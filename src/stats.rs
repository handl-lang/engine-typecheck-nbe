//! Typecheck-path instrumentation (phase-3 WS-H perf pass).
//!
//! Process-global counters for the conversion/unification and
//! re-synthesis hot paths, gated on `HNL_TC_STATS` — when the env var is
//! unset every probe is a single relaxed `AtomicBool` load, so the
//! counters are free to keep in-tree permanently. `HNL_TC_STATS=1 cargo
//! test -- --nocapture` prints the report from the `zzz_tc_stats` probe
//! test (and any test can read [`report`]).

use rustc_hash::FxHasher;
use serde_json::Value;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

static ON: OnceLock<bool> = OnceLock::new();

/// Whether instrumentation is collecting (`HNL_TC_STATS` set truthy).
#[inline]
pub fn enabled() -> bool {
    *ON.get_or_init(|| {
        std::env::var("HNL_TC_STATS")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false)
    })
}

// --- unify / conv pair probes -------------------------------------------
// `(fp(a), fp(b))` fingerprints seen per entry point — a repeated pair is
// a re-derived conversion a memo would have skipped.

pub static DU_CALLS: AtomicU64 = AtomicU64::new(0);
pub static DU_DUPS: AtomicU64 = AtomicU64::new(0);
pub static U_CALLS: AtomicU64 = AtomicU64::new(0);
pub static U_DUPS: AtomicU64 = AtomicU64::new(0);
pub static UI_CALLS: AtomicU64 = AtomicU64::new(0);
pub static UI_DUPS: AtomicU64 = AtomicU64::new(0);
/// `dep_unify` calls where the post-zonk pair is already `a == b` — the
/// cheap early-return share (memoization cannot beat it).
pub static DU_EQ_FAST: AtomicU64 = AtomicU64::new(0);

static DU_SEEN: Mutex<Option<HashSet<(u64, u64)>>> = Mutex::new(None);
static U_SEEN: Mutex<Option<HashSet<(u64, u64)>>> = Mutex::new(None);
static UI_SEEN: Mutex<Option<HashSet<(u64, u64)>>> = Mutex::new(None);

// --- re-synthesis probes --------------------------------------------------
// Same-node/same-ctor revisits — `dep_synth_type`/`synth_data`/
// `instantiate_ctor_full` re-deriving what a cache would have returned.

pub static DST_CALLS: AtomicU64 = AtomicU64::new(0);
pub static DST_DUPS: AtomicU64 = AtomicU64::new(0);
pub static SD_CALLS: AtomicU64 = AtomicU64::new(0);
pub static SD_DUPS: AtomicU64 = AtomicU64::new(0);
pub static IC_CALLS: AtomicU64 = AtomicU64::new(0);
pub static IC_DUPS: AtomicU64 = AtomicU64::new(0);
pub static NIT_CALLS: AtomicU64 = AtomicU64::new(0);
pub static NIT_DUPS: AtomicU64 = AtomicU64::new(0);
pub static EVAL_INDEX_CALLS: AtomicU64 = AtomicU64::new(0);
/// How many `instantiate_ctor_full` calls minted a fresh `type.function`
/// chain vs how many ctor names are involved (re-instantiation share).
static IC_SEEN: Mutex<Option<HashSet<String>>> = Mutex::new(None);

static DST_SEEN: Mutex<Option<HashSet<usize>>> = Mutex::new(None);
static SD_SEEN: Mutex<Option<HashSet<usize>>> = Mutex::new(None);
static NIT_SEEN: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

// --- inclusive timers (ns) -------------------------------------------------
// Where the checker's time actually goes — one probe at fn entry, an RAII
// drop accumulates the elapsed nanos.

pub static T_SYNTH_NODE: AtomicU64 = AtomicU64::new(0);
pub static T_CHECK_NODE: AtomicU64 = AtomicU64::new(0);
pub static T_EVAL_SCHEME: AtomicU64 = AtomicU64::new(0);
pub static T_DEP_UNIFY: AtomicU64 = AtomicU64::new(0);
pub static T_UNIFY: AtomicU64 = AtomicU64::new(0);
pub static T_UNIFY_INDEX: AtomicU64 = AtomicU64::new(0);
pub static T_DEP_SYNTH_TYPE: AtomicU64 = AtomicU64::new(0);
pub static T_INSTANTIATE_CTOR: AtomicU64 = AtomicU64::new(0);
pub static T_EVAL_INDEX: AtomicU64 = AtomicU64::new(0);
pub static T_NODE_INDEX_OR_TYPE: AtomicU64 = AtomicU64::new(0);
pub static T_ZONK: AtomicU64 = AtomicU64::new(0);

/// RAII inclusive timer — `let _t = stats::timer(&T_X)` at fn entry.
pub struct Probe(std::time::Instant, &'static AtomicU64);

impl Drop for Probe {
    fn drop(&mut self) {
        self.1
            .fetch_add(self.0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

#[inline]
pub fn timer(counter: &'static AtomicU64) -> Option<Probe> {
    enabled().then(|| Probe(std::time::Instant::now(), counter))
}

/// Canonical structural fingerprint of a tagged type `Value` —
/// object keys sort so equal values hash equal regardless of
/// insertion order.
pub fn vfp(v: &Value) -> u64 {
    let mut h = FxHasher::default();
    vfp_into(v, &mut h);
    h.finish()
}

fn vfp_into(v: &Value, h: &mut FxHasher) {
    match v {
        Value::Null => 0u8.hash(h),
        Value::Bool(b) => (1u8, b).hash(h),
        Value::Number(n) => (2u8, n.to_string()).hash(h),
        Value::String(s) => (3u8, s).hash(h),
        Value::Array(xs) => {
            4u8.hash(h);
            xs.len().hash(h);
            for x in xs {
                vfp_into(x, h);
            }
        }
        Value::Object(m) => {
            5u8.hash(h);
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            keys.len().hash(h);
            for k in keys {
                k.hash(h);
                vfp_into(&m[k], h);
            }
        }
    }
}

/// Record a `(a, b)` unification pair; returns `true` when this exact
/// structural pair was already probed at the same entry point.
fn seen_pair(seen: &Mutex<Option<HashSet<(u64, u64)>>>, a: &Value, b: &Value) -> bool {
    let key = (vfp(a), vfp(b));
    let mut guard = seen.lock().unwrap();
    guard.get_or_insert_with(HashSet::new).insert(key)
}

fn seen_key(seen: &Mutex<Option<HashSet<usize>>>, k: usize) -> bool {
    let mut guard = seen.lock().unwrap();
    guard.get_or_insert_with(HashSet::new).insert(k)
}

/// Probe at `dep_unify` entry (post-zonk pair).
#[inline]
pub fn dep_unify(a: &Value, b: &Value) {
    if !enabled() {
        return;
    }
    DU_CALLS.fetch_add(1, Ordering::Relaxed);
    if a == b {
        DU_EQ_FAST.fetch_add(1, Ordering::Relaxed);
    } else if seen_pair(&DU_SEEN, a, b) {
        DU_DUPS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Probe at `unify` entry.
#[inline]
pub fn unify(a: &Value, b: &Value) {
    if !enabled() {
        return;
    }
    U_CALLS.fetch_add(1, Ordering::Relaxed);
    if a != b && seen_pair(&U_SEEN, a, b) {
        U_DUPS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Probe at `unify_index` entry.
#[inline]
pub fn unify_index(a: &Value, b: &Value) {
    if !enabled() {
        return;
    }
    UI_CALLS.fetch_add(1, Ordering::Relaxed);
    if a != b && seen_pair(&UI_SEEN, a, b) {
        UI_DUPS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Probe at `dep_synth_type` entry (`idx` = AST node index).
#[inline]
pub fn dep_synth_type(idx: usize) {
    if !enabled() {
        return;
    }
    DST_CALLS.fetch_add(1, Ordering::Relaxed);
    if seen_key(&DST_SEEN, idx) {
        DST_DUPS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Probe at `synth_data` entry (`idx` = AST node index).
#[inline]
pub fn synth_data(idx: usize) {
    if !enabled() {
        return;
    }
    SD_CALLS.fetch_add(1, Ordering::Relaxed);
    if seen_key(&SD_SEEN, idx) {
        SD_DUPS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Probe at `instantiate_ctor_full` entry (`ctor` = ctor name).
#[inline]
pub fn instantiate_ctor(ctor: &str) {
    if !enabled() {
        return;
    }
    IC_CALLS.fetch_add(1, Ordering::Relaxed);
    let mut guard = IC_SEEN.lock().unwrap();
    if !guard
        .get_or_insert_with(HashSet::new)
        .insert(ctor.to_string())
    {
        IC_DUPS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Probe at `node_index_or_type` entry (`idx` = AST node index).
#[inline]
pub fn node_index_or_type(idx: usize) {
    if !enabled() {
        return;
    }
    NIT_CALLS.fetch_add(1, Ordering::Relaxed);
    if seen_key(&NIT_SEEN, idx) {
        NIT_DUPS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Probe at `eval_index` entry.
#[inline]
pub fn eval_index() {
    if enabled() {
        EVAL_INDEX_CALLS.fetch_add(1, Ordering::Relaxed);
    }
}

fn pct(dups: u64, calls: u64) -> String {
    if calls == 0 {
        "—".to_string()
    } else {
        format!("{:.1}%", (dups as f64) * 100.0 / (calls as f64))
    }
}

/// One-line-per-counter report for `eprintln!`.
pub fn report() -> String {
    let (du, dud, due) = (
        DU_CALLS.load(Ordering::Relaxed),
        DU_DUPS.load(Ordering::Relaxed),
        DU_EQ_FAST.load(Ordering::Relaxed),
    );
    let (u, ud) = (
        U_CALLS.load(Ordering::Relaxed),
        U_DUPS.load(Ordering::Relaxed),
    );
    let (ui, uid) = (
        UI_CALLS.load(Ordering::Relaxed),
        UI_DUPS.load(Ordering::Relaxed),
    );
    let (dst, dstd) = (
        DST_CALLS.load(Ordering::Relaxed),
        DST_DUPS.load(Ordering::Relaxed),
    );
    let (sd, sdd) = (
        SD_CALLS.load(Ordering::Relaxed),
        SD_DUPS.load(Ordering::Relaxed),
    );
    let (ic, icd) = (
        IC_CALLS.load(Ordering::Relaxed),
        IC_DUPS.load(Ordering::Relaxed),
    );
    let (nit, nitd) = (
        NIT_CALLS.load(Ordering::Relaxed),
        NIT_DUPS.load(Ordering::Relaxed),
    );
    let ei = EVAL_INDEX_CALLS.load(Ordering::Relaxed);
    let (cc, cfp, cdr, cdf) = handl_value::nbe::conv_stats();
    let ms = |c: &AtomicU64| format!("{:.1}ms", c.load(Ordering::Relaxed) as f64 / 1e6);
    format!(
        "tc_stats: dep_unify {du} calls ({dud} dup pairs = {}, {due} eq-fast) \
         | unify {u} ({ud} dup = {}) | unify_index {ui} ({uid} dup = {}) \
         | dep_synth_type {dst} ({dstd} node revisits = {}) \
         | synth_data {sd} ({sdd} revisits) \
         | instantiate_ctor {ic} ({icd} ctor re-instantiations = {}) \
         | node_index_or_type {nit} ({nitd} revisits = {}) \
         | eval_index {ei} \
         | nbe::conv {cc} ({cfp} ptr-fast, {cdr} rigid dups = {}, {cdf} flex dups)\n\
         tc_time: synth_node {} check_node {} eval_scheme {} dep_unify {} unify {} \
         unify_index {} dep_synth_type {} instantiate_ctor {} eval_index {} \
         node_index_or_type {} zonk {}",
        pct(dud, du),
        pct(ud, u),
        pct(uid, ui),
        pct(dstd, dst),
        pct(icd, ic),
        pct(nitd, nit),
        pct(cdr, cc),
        ms(&T_SYNTH_NODE),
        ms(&T_CHECK_NODE),
        ms(&T_EVAL_SCHEME),
        ms(&T_DEP_UNIFY),
        ms(&T_UNIFY),
        ms(&T_UNIFY_INDEX),
        ms(&T_DEP_SYNTH_TYPE),
        ms(&T_INSTANTIATE_CTOR),
        ms(&T_EVAL_INDEX),
        ms(&T_NODE_INDEX_OR_TYPE),
        ms(&T_ZONK),
    )
}

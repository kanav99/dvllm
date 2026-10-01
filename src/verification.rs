#![allow(static_mut_refs)]

use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
// Real secp256k1 (not ark_test_curves, which is unoptimized and explicitly
// meant for testing only). This group has a prime order `r`, and is
// associated with a prime field `Fr`.
use ark_secp256k1::{Projective as G, Affine as GAffine, Fr as ScalarField};
use ark_std::UniformRand;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use bincode;
use std::thread;
// use std::time::Instant;

use std::ffi::c_uint;
// Sized to the largest dimension any supported model's commitments need to
// index (HAZMAT/GLOBAL_SETUP are indexed up to the classifier's vocab_size
// rows). Was 128256 (Llama 3.2's vocab size); Gemma 3 needs 262144.
// const MAX_DIM: usize = 262144;

#[no_mangle]
extern "C" fn hello_rust() {
    println!("Hello from Rust!");
}

static mut HAZMAT: Vec<ScalarField> = Vec::new();
static mut GLOBAL_SETUP: Vec<GAffine> = Vec::new();
// Multiple commits can be staged at once (one per (layer, ID) pair), so that
// every commit file needed for a verification pass can be read from disk
// up front, before any of the MSM/compare work (rust_mcs_verify) runs.
static mut COMMITS: Vec<(i32, i32, Vec<GAffine>)> = Vec::new();

#[no_mangle]
extern "C" fn rust_mcs_gen_setup(fname: *const i8, max_dim: usize) // -> Vec<GAffine>
{
    let mut rng = ark_std::test_rng();
    let g = GAffine::generator();
    unsafe {
        GLOBAL_SETUP.clear();
        HAZMAT.clear();
    }

    // generate a list of `max_dim` random group elements
    let mut setup = Vec::new();
    for _ in 0..max_dim {
        let r = ScalarField::rand(&mut rng);
        let pt = (g * r).into_affine();
        unsafe {
            HAZMAT.push(r);
            GLOBAL_SETUP.push(pt);
        }
        let mut compressed_bytes = Vec::new();
        pt.serialize_compressed(&mut compressed_bytes).unwrap();
        setup.push(compressed_bytes);
    }
    
    // dump the setup to a file
    let setup_bytes = bincode::serialize(&setup).unwrap();
    let fname_str = unsafe { std::ffi::CStr::from_ptr(fname).to_string_lossy().into_owned() };
    std::fs::write(&fname_str, &setup_bytes).unwrap();
}

#[no_mangle]
extern "C" fn rust_mcs_load_setup(fname: *const i8)
{
    let fname_str = unsafe { std::ffi::CStr::from_ptr(fname).to_string_lossy().into_owned() };
    let setup_bytes = std::fs::read(&fname_str).unwrap();
    let setup: Vec<Vec<u8>> = bincode::deserialize(&setup_bytes).unwrap();
    let max_dim = setup.len();
    unsafe {
        GLOBAL_SETUP.clear();
    }
    for i in 0..max_dim {
        let pt = GAffine::deserialize_compressed(&setup[i][..]).unwrap();
        unsafe {
            GLOBAL_SETUP.push(pt);
            assert!(GLOBAL_SETUP[i].is_on_curve());
        }
    }
}

#[no_mangle]
extern "C" fn rust_mcs_gen_commit(_d0: c_uint, _d1: c_uint, _m: *const i32, fname: *const i8)
{
    let d0 = _d0 as usize; // outsize, num rows
    let d1 = _d1 as usize; // insize, num cols
    let m = unsafe { std::slice::from_raw_parts(_m, d0 * d1) };
    let max_dim = unsafe { GLOBAL_SETUP.len() };
    assert!(d1 < max_dim);
    unsafe {
        assert!(HAZMAT.len() != 0);
    }

    let mut _commit = Vec::new();
    for j in 0..d1 {
        unsafe {
            let mut a = ScalarField::from(0);
            for i in 0..d0 {
                a = a + HAZMAT[i] * ScalarField::from(m[i * d1 + j]);
            }
            let r = GAffine::generator() * a;
            let mut compressed_bytes = Vec::new();
            r.serialize_compressed(&mut compressed_bytes).unwrap();
            _commit.push(compressed_bytes);
        }
    }

    // dump the commit to a file
    let commit_bytes = bincode::serialize(&_commit).unwrap();
    let fname_str = unsafe { std::ffi::CStr::from_ptr(fname).to_string_lossy().into_owned() };
    std::fs::write(&fname_str, &commit_bytes).unwrap();
}

#[no_mangle]
extern "C" fn rust_mcs_clear_commits()
{
    unsafe {
        COMMITS.clear();
    }
}

// Loads one commit file and stages it under (layer, id), alongside whatever
// commits are already staged. Call this once per (layer, id) pair needed by
// a verification pass, all before calling rust_mcs_verify.
#[no_mangle]
extern "C" fn rust_mcs_load_commit(layer: i32, id: i32, fname: *const i8)
{
    let fname_str = unsafe { std::ffi::CStr::from_ptr(fname).to_string_lossy().into_owned() };
    let commit_bytes = std::fs::read(&fname_str).unwrap();
    let commit: Vec<Vec<u8>> = bincode::deserialize(&commit_bytes).unwrap();
    let mut points = Vec::with_capacity(commit.len());
    for c in commit.iter() {
        let r = GAffine::deserialize_compressed(&c[..]).unwrap();
        assert!(r.is_on_curve());
        points.push(r);
    }
    unsafe {
        COMMITS.push((layer, id, points));
    }
}

#[no_mangle]
extern "C" fn rust_mcs_verify(layer: i32, id: i32, _d0: c_uint, _d1: c_uint, _x: *const i128, _y: *const i128) -> c_uint
{
    let d0 = _d0 as usize;
    let d1 = _d1 as usize;
    let x = unsafe { std::slice::from_raw_parts(_x, d1) };
    let y = unsafe { std::slice::from_raw_parts(_y, d0) };
    let x_field = x.iter().map(|&x| ScalarField::from(x)).collect::<Vec<_>>();
    let y_field = y.iter().map(|&y| ScalarField::from(y)).collect::<Vec<_>>();
    let max_dim = unsafe { GLOBAL_SETUP.len() };
    assert!(d1 < max_dim);

    // Raw-pointer indirection so the borrow below is inferred 'static (same
    // reasoning as indexing GLOBAL_SETUP directly): COMMITS is staged in
    // full by rust_mcs_load_commit before any rust_mcs_verify call runs, so
    // there's no concurrent mutation to race with these reads.
    let commit_ptr: *const Vec<GAffine> = unsafe {
        COMMITS
            .iter()
            .find(|(l, i, _)| *l == layer && *i == id)
            .map(|(_, _, v)| v as *const Vec<GAffine>)
            .expect("commit not staged for (layer, id); call rust_mcs_load_commit first")
    };
    let commit_ref: &'static Vec<GAffine> = unsafe { &*commit_ptr };
    let commit_points: &'static [GAffine] = &commit_ref[..];
    assert!(commit_points.len() == d1);

    unsafe {
        let setup_points: &'static [GAffine] = &GLOBAL_SETUP[..d0];
        let t1 = thread::spawn(move || G::msm(setup_points, &y_field).unwrap());
        let t2 = thread::spawn(move || G::msm(commit_points, &x_field).unwrap());
        if t1.join().unwrap() == t2.join().unwrap() {
            return 1;
        } else {
            return 0;
        }
    }
}

// Verifies every (layer, id) pair in one shot: instead of n separate MSMs
// per side, builds one concatenated point/scalar list per side and runs a
// single MSM per side. MSM cost grows roughly like N/log(N), so one MSM
// over the concatenation of n groups is cheaper than n separate MSMs over
// the same total number of points.
//
// This concatenates with coefficient 1 per id, no extra per-id weighting.
// That's still sound: each x/y pair fed in here is already
// xs[id] = sum_k coeff_k * x_k (see rust_mcs_load_commit's caller,
// Verifier::Eat in verification.hpp), where coeff_k is a fresh random value
// the verifier draws *after* it has received call k's (x_k, y_k) from the
// (untrusted) party being checked. Expanding the check across every id and
// every underlying Eat() call, the combined discrepancy is
// sum_{id,k} coeff_{id,k} * (e_{id,k} . r) for per-call errors e_{id,k} and
// the secret trusted-setup exponents r -- a sum of independently, freshly
// randomized terms regardless of how many ids are concatenated or with what
// fixed weight. So concatenating whole ids together with weight 1 doesn't
// reopen any cross-id cancellation: the per-call randomness in Eat() already
// covers it.
#[no_mangle]
extern "C" fn rust_mcs_verify_batch(
    _n: c_uint,
    layers: *const i32,
    ids: *const i32,
    _d0s: *const c_uint,
    _d1s: *const c_uint,
    x_concat: *const i128,
    y_concat: *const i128,
) -> c_uint
{
    let n = _n as usize;
    if n == 0 {
        return 1;
    }
    let layers = unsafe { std::slice::from_raw_parts(layers, n) };
    let ids = unsafe { std::slice::from_raw_parts(ids, n) };
    let d0s = unsafe { std::slice::from_raw_parts(_d0s, n) };
    let d1s = unsafe { std::slice::from_raw_parts(_d1s, n) };

    let total_x: usize = d1s.iter().map(|&d| d as usize).sum();
    let total_y: usize = d0s.iter().map(|&d| d as usize).sum();
    let x_concat = unsafe { std::slice::from_raw_parts(x_concat, total_x) };
    let y_concat = unsafe { std::slice::from_raw_parts(y_concat, total_y) };

    let max_dim = unsafe { GLOBAL_SETUP.len() };

    let mut lhs_points: Vec<GAffine> = Vec::with_capacity(total_x);
    let mut lhs_scalars: Vec<ScalarField> = Vec::with_capacity(total_x);
    let mut rhs_points: Vec<GAffine> = Vec::with_capacity(total_y);
    let mut rhs_scalars: Vec<ScalarField> = Vec::with_capacity(total_y);

    let mut x_off = 0usize;
    let mut y_off = 0usize;
    for i in 0..n {
        let d0 = d0s[i] as usize;
        let d1 = d1s[i] as usize;
        assert!(d1 < max_dim);

        let commit_ptr: *const Vec<GAffine> = unsafe {
            COMMITS
                .iter()
                .find(|(l, id, _)| *l == layers[i] && *id == ids[i])
                .map(|(_, _, v)| v as *const Vec<GAffine>)
                .expect("commit not staged for (layer, id); call rust_mcs_load_commit first")
        };
        let commit_ref: &Vec<GAffine> = unsafe { &*commit_ptr };
        assert!(commit_ref.len() == d1);

        for j in 0..d1 {
            lhs_points.push(commit_ref[j].clone());
            lhs_scalars.push(ScalarField::from(x_concat[x_off + j]));
        }
        for j in 0..d0 {
            rhs_points.push(unsafe { GLOBAL_SETUP[j].clone() });
            rhs_scalars.push(ScalarField::from(y_concat[y_off + j]));
        }

        x_off += d1;
        y_off += d0;
    }

    let t1 = thread::spawn(move || G::msm(&rhs_points, &rhs_scalars).unwrap());
    let t2 = thread::spawn(move || G::msm(&lhs_points, &lhs_scalars).unwrap());
    if t1.join().unwrap() == t2.join().unwrap() {
        1
    } else {
        0
    }
}

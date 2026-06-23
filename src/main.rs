/*
NN Step 12. Dense Layer Backward
[src/solutions/nn/12-DenseBackward.rs]


Goal:
Implement the backward pass for a single dense (fully-connected) layer.


Forward recap:
A dense layer without activation computes:

  Z = X * W + b

where:
- X has shape (batch_size x in_dim)
- W has shape (in_dim x out_dim)
- b has shape (out_dim)
- Z has shape (batch_size x out_dim)


Backward goal:
Given:
- the input X
- the weights W
- and the upstream gradient dL_dZ (same shape as Z)

compute:
- dL_dW: gradient of loss w.r.t. weights W
- dL_db: gradient of loss w.r.t. bias b
- dL_dX: gradient of loss w.r.t. inputs X


Shape rules:
Let:
- X: (B x I)
- W: (I x O)
- Z: (B x O)
- dL_dZ: (B x O)

Then:
- dL_dW: (I x O)
- dL_db: (O)
- dL_dX: (B x I)


Definitions (matrix form):
Using standard batched dense-layer backprop identities:[web:128][web:129]

- dL_dW = X^T · dL_dZ
- dL_db = sum over batch of dL_dZ
- dL_dX = dL_dZ · W^T

More explicitly:

  dL_dW[i][j] = sum_{k=0..B-1} X[k][i] * dL_dZ[k][j]

  dL_db[j]    = sum_{k=0..B-1} dL_dZ[k][j]

  dL_dX[k][i] = sum_{j=0..O-1} dL_dZ[k][j] * W[i][j]


Notes:
- This problem is ONLY the linear (affine) part: Z = XW + b.
- Any activation (like ReLU) is handled separately.
- Assume shapes are valid; no need for defensive checks in this exercise.


Examples (scalar intuition):

If batch_size = 1, in_dim = 1, out_dim = 1:

Let:
  X = [[x]]
  W = [[w]]
  b = [b]
  Z = [[x * w + b]]

Given upstream gradient:
  dL_dZ = [[g]]

Then:
  dL_dW = [[x * g]]
  dL_db = [g]
  dL_dX = [[w * g]]


In matrix form, for small sizes, the formulas above reduce exactly to these scalar cases.
*/


struct Solution;


pub struct DenseGrads {
    pub dW: Vec<Vec<f32>>,  // same shape as W: (in_dim x out_dim)
    pub db: Vec<f32>,       // same shape as b: (out_dim)
    pub dX: Vec<Vec<f32>>,  // same shape as X: (batch_size x in_dim)
}


impl Solution {
    pub fn dense_backward(
        x: Vec<Vec<f32>>,      // (B x I)
        w: Vec<Vec<f32>>,      // (I x O)
        dL_dZ: Vec<Vec<f32>>,  // (B x O)
    ) -> DenseGrads {
        // TODO: implement
        // - Compute dW = X^T · dL_dZ
        // - Compute db = sum over batch of dL_dZ
        // - Compute dX = dL_dZ · W^T

        // You can either:
        // - use your existing transpose + matmul helpers, OR
        // - write the loops directly here.

        /* Action plan
        Let's do it this way. We're going to rebuild my old functions and maybe add some small tweaks based on my new learnings/preferences.

        This time around, I'm not going to delete the hints because I think they are going to be kinda necessary.
        
        We'll try without first and then I'll update this later. Then later on I'll update my dev journal.
        */

        let dW = 

        DenseGrads {
            dW: vec![],
            db: vec![],
            dX: vec![],
        }
    }
}


// Tests


fn matrices_close(a: &[Vec<f32>], b: &[Vec<f32>], eps: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for i in 0..a.len() {
        if a[i].len() != b[i].len() {
            return false;
        }
        for j in 0..a[i].len() {
            if (a[i][j] - b[i][j]).abs() > eps {
                return false;
            }
        }
    }
    true
}

fn vectors_close(a: &[f32], b: &[f32], eps: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for i in 0..a.len() {
        if (a[i] - b[i]).abs() > eps {
            return false;
        }
    }
    true
}


fn print_result(
    case_name: &str,
    x: Vec<Vec<f32>>,
    w: Vec<Vec<f32>>,
    dL_dZ: Vec<Vec<f32>>,
    expected_dW: Vec<Vec<f32>>,
    expected_db: Vec<f32>,
    expected_dX: Vec<Vec<f32>>,
) {
    let grads = Solution::dense_backward(x.clone(), w.clone(), dL_dZ.clone());

    let dW_ok = matrices_close(&grads.dW, &expected_dW, 1e-4);
    let db_ok = vectors_close(&grads.db, &expected_db, 1e-4);
    let dX_ok = matrices_close(&grads.dX, &expected_dX, 1e-4);

    let passed = dW_ok && db_ok && dX_ok;

    let green = "\x1b[32m";
    let red = "\x1b[31m";
    let cyan = "\x1b[36m";
    let yellow = "\x1b[33m";
    let bold = "\x1b[1m";
    let reset = "\x1b[0m";

    println!("\n{}{}{}{}", bold, cyan, case_name, reset);
    println!("{}X:{}      {:?}", yellow, reset, x);
    println!("{}W:{}      {:?}", yellow, reset, w);
    println!("{}dL_dZ:{}  {:?}", yellow, reset, dL_dZ);
    println!("{}Expected dW:{} {:?}", yellow, reset, expected_dW);
    println!("{}Actual dW:{}   {:?}", yellow, reset, grads.dW);
    println!("{}Expected db:{} {:?}", yellow, reset, expected_db);
    println!("{}Actual db:{}   {:?}", yellow, reset, grads.db);
    println!("{}Expected dX:{} {:?}", yellow, reset, expected_dX);
    println!("{}Actual dX:{}   {:?}", yellow, reset, grads.dX);

    if passed {
        println!("{}✅ PASS{}", green, reset);
    } else {
        println!("{}❌ FAIL{}", red, reset);
    }
}


fn main() {
    println!("\x1b[1m\x1b[35mNN Runner: Dense Backward\x1b[0m");

    // Test 1: scalar case (B=1, I=1, O=1)
    // X = [[2.0]], W = [[3.0]], dL_dZ = [[4.0]]
    // dW = [[2*4]] = [[8.0]]
    // db = [4.0]
    // dX = [[3*4]] = [[12.0]]
    print_result(
        "Test Case 1: scalar dense layer",
        vec![vec![2.0]],
        vec![vec![3.0]],
        vec![vec![4.0]],
        vec![vec![8.0]],
        vec![4.0],
        vec![vec![12.0]],
    );

    // Test 2: small batched case (B=2, I=2, O=1)
    // X = [[1, 2],
    //      [3, 4]]
    // W = [[5],
    //      [6]]
    // dL_dZ = [[0.1],
    //          [0.2]]
    //
    // dW = X^T · dL_dZ:
    // X^T = [[1, 3],
    //        [2, 4]]
    //
    // dW[0][0] = 1*0.1 + 3*0.2 = 0.7
    // dW[1][0] = 2*0.1 + 4*0.2 = 1.0
    //
    // db[0] = 0.1 + 0.2 = 0.3
    //
    // dX = dL_dZ · W^T:
    // W^T = [[5, 6]]
    //
    // For first sample:
    //  dX[0][0] = 0.1*5 = 0.5
    //  dX[0][1] = 0.1*6 = 0.6
    //
    // For second sample:
    //  dX[1][0] = 0.2*5 = 1.0
    //  dX[1][1] = 0.2*6 = 1.2
    print_result(
        "Test Case 2: small batched dense layer",
        vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        vec![vec![5.0], vec![6.0]],
        vec![vec![0.1], vec![0.2]],
        vec![vec![0.7], vec![1.0]],
        vec![0.3],
        vec![vec![0.5, 0.6], vec![1.0, 1.2]],
    );

    // Test 3: zero upstream gradient → zero grads
    print_result(
        "Test Case 3: zero upstream gradient",
        vec![vec![1.0, -2.0], vec![3.0, 4.0]],
        vec![vec![0.5, -1.0], vec![2.0, 3.0]],
        vec![vec![0.0, 0.0], vec![0.0, 0.0]],
        vec![vec![0.0, 0.0], vec![0.0, 0.0]],
        vec![0.0, 0.0],
        vec![vec![0.0, 0.0], vec![0.0, 0.0]],
    );
}
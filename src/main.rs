/*
NN Step 10. Backprop for One Dense Layer
[src/solutions/nn/10-DenseBackward.rs]

Goal:
Implement the backward pass for a single dense (fully-connected) layer.

Forward recap:
Consider a dense layer without activation (or with activation handled elsewhere):

- Input X has shape (batch_size, in_dim)
- Weights W have shape (in_dim, out_dim)
- Bias b has shape (out_dim)

The forward pass computes:
Y = X · W + b

Here:
- Y has shape (batch_size, out_dim)
- "·" is matrix multiplication
- The bias b is added to each row of X·W

Backward setup:
Assume you are given:

- X: the original input to the layer, shape (batch_size, in_dim)
- W: the layer weights, shape (in_dim, out_dim)
- dY: the gradient of the loss with respect to Y, shape (batch_size, out_dim)

Your task:
Compute the following three gradients:

- dX: gradient of the loss w.r.t. the inputs X, shape (batch_size, in_dim)
- dW: gradient of the loss w.r.t. the weights W, shape (in_dim, out_dim)
- db: gradient of the loss w.r.t. the bias b, shape (out_dim)

Use only:
- Your transpose function
- Your matmul
- Elementwise operations and sums

You should rely on the matrix calculus / backprop view:
“how much does nudging each entry of X, W, and b change the loss?”,
given dY from the next level up.[web:799][web:743]

Hints about *what* to use (but not the explicit formulas):
- Think about how Y changes if you tweak X while holding W fixed.
- Think about how Y changes if you tweak W while holding X fixed.
- Think about how Y changes if you tweak b.
- Then apply the chain rule with dY.[web:799][web:800]

You can verify your shapes by hand before coding:
- X is (B, In)
- W is (In, Out)
- Y is (B, Out)
- dY is (B, Out)
- dX, dW, db must each have the shapes listed above.

No shape checking is required; assume the inputs make sense.
Focus on the math and using your existing building blocks.

Example 1 (tiny batch and dims, conceptual):
If batch_size = 1, in_dim = 2, out_dim = 1, this becomes scalar/2D math
that should match the simple hand-derived formulas for a single neuron
with two inputs.[web:799]

Example 2:
If dY is all zeros, all gradients (dX, dW, db) must be zero as well.
*/

struct Solution;

// Assume you already implemented these elsewhere in your project;
// here we include minimal signatures so this file compiles when integrated.

fn matmul(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let rows = a.len();
    let mid = b.len();
    let cols = b[0].len();
    let mut out = vec![vec![0.0; cols]; rows];
    for i in 0..rows {
        for k in 0..mid {
            for j in 0..cols {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

fn transpose(a: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let rows = a.len();
    let cols = a[0].len();
    let mut out = vec![vec![0.0; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j][i] = a[i][j];
        }
    }
    out
}

impl Solution {
    pub fn dense_backward(
        x: Vec<Vec<f32>>,
        w: Vec<Vec<f32>>,
        dy: Vec<Vec<f32>>,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>) {
        let batch = x.len();
        let in_dim = x[0].len();
        let out_dim = w[0].len();

        // dX shape: (batch, in_dim)
        // dW shape: (in_dim, out_dim)
        // db shape: (out_dim)

        // Implement using matmul, transpose, and elementwise sums.

        // dX = ?
        // dW = ?
        // db = ?

        let dx = vec![vec![0.0; in_dim]; batch];
        let dw = vec![vec![0.0; out_dim]; in_dim];
        let db = vec![0.0; out_dim];

        (dx, dw, db)
    }
}

// Helpers and tests

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
    dy: Vec<Vec<f32>>,
    expected_dx: Vec<Vec<f32>>,
    expected_dw: Vec<Vec<f32>>,
    expected_db: Vec<f32>,
) {
    let (dx, dw, db) = Solution::dense_backward(x.clone(), w.clone(), dy.clone());

    let passed_dx = matrices_close(&dx, &expected_dx, 1e-5);
    let passed_dw = matrices_close(&dw, &expected_dw, 1e-5);
    let passed_db = vectors_close(&db, &expected_db, 1e-5);

    let green = "\x1b[32m";
    let red = "\x1b[31m";
    let cyan = "\x1b[36m";
    let yellow = "\x1b[33m";
    let bold = "\x1b[1m";
    let reset = "\x1b[0m";

    println!("\n{}{}{}{}", bold, cyan, case_name, reset);
    println!("{}X:{}   {:?}", yellow, reset, x);
    println!("{}W:{}   {:?}", yellow, reset, w);
    println!("{}dY:{}  {:?}", yellow, reset, dy);
    println!("{}Expected dX:{} {:?}", yellow, reset, expected_dx);
    println!("{}Actual   dX:{} {:?}", yellow, reset, dx);
    println!("{}Expected dW:{} {:?}", yellow, reset, expected_dw);
    println!("{}Actual   dW:{} {:?}", yellow, reset, dw);
    println!("{}Expected db:{} {:?}", yellow, reset, expected_db);
    println!("{}Actual   db:{} {:?}", yellow, reset, db);

    if passed_dx && passed_dw && passed_db {
        println!("{}✅ PASS{}", green, reset);
    } else {
        println!("{}❌ FAIL{}", red, reset);
    }
}

fn main() {
    println!("\x1b[1m\x1b[35mNN Runner: Dense Layer Backward\x1b[0m");

    // Test Case 1: batch_size=1, in_dim=2, out_dim=1
    // X: [[x1, x2]]
    // W: [[w11],
    //     [w21]]
    // Y = [[x1*w11 + x2*w21 + b1]]
    // Let dY = [[g]].
    // Hand-derivable scalar-case gradients:
    // dX1 = g * w11, dX2 = g * w21
    // dW11 = g * x1, dW21 = g * x2
    // db1 = g
    let x1 = vec![vec![1.0, 2.0]];
    let w1 = vec![vec![3.0], vec![4.0]];
    let dy1 = vec![vec![0.5]];

    let expected_dx1 = vec![vec![0.5 * 3.0, 0.5 * 4.0]];
    let expected_dw1 = vec![vec![0.5 * 1.0], vec![0.5 * 2.0]];
    let expected_db1 = vec![0.5];

    print_result(
        "Test Case 1 (scalar neuron)",
        x1,
        w1,
        dy1,
        expected_dx1,
        expected_dw1,
        expected_db1,
    );

    // Test Case 2: batch_size=2, in_dim=2, out_dim=2
    // Just a small numeric example that should match
    // the matrix calculus you implement.
    let x2 = vec![vec![1.0, 2.0], vec![3.0, 4.0]]; // shape (2,2)
    let w2 = vec![vec![1.0, 0.0], vec![0.0, 1.0]]; // shape (2,2), identity-ish
    let dy2 = vec![vec![1.0, -1.0], vec![0.5, 0.5]]; // shape (2,2)

    // You can compute these by hand using your formulas.
    // I’m giving explicit numbers here so the runner is fully self-contained
    // once you've done that exercise.

    // dX = dY · W^T
    let expected_dx2 = matmul(&dy2, &transpose(&w2));

    // dW = X^T · dY
    let expected_dw2 = matmul(&transpose(&x2), &dy2);

    // db = sum over batch of dY
    let mut expected_db2 = vec![0.0; 2];
    for i in 0..dy2.len() {
        for j in 0..2 {
            expected_db2[j] += dy2[i][j];
        }
    }

    print_result(
        "Test Case 2 (2x2 batch)",
        x2,
        w2,
        dy2,
        expected_dx2,
        expected_dw2,
        expected_db2,
    );

    // Test Case 3: dY is all zeros -> all gradients are zero
    let x3 = vec![vec![1.0, -1.0], vec![2.0, 0.5]];
    let w3 = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
    let dy3 = vec![vec![0.0, 0.0], vec![0.0, 0.0]];

    let expected_dx3 = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
    let expected_dw3 = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
    let expected_db3 = vec![0.0, 0.0];

    print_result(
        "Test Case 3 (zero upstream grad)",
        x3,
        w3,
        dy3,
        expected_dx3,
        expected_dw3,
        expected_db3,
    );
}
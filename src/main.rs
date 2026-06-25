/*
NN Step 13. Dense Layer Backward
[src/solutions/nn/13-DenseBackward.rs]


Goal:
Implement the backward pass for a single dense (fully-connected) layer.


Why this matters for neural networks:
A dense layer is one of the core building blocks of neural networks.
In the forward pass, it transforms inputs using weights and bias:


  Z = X * W + b


To train the network, we need to backpropagate through this operation.
That means:
- finding how the loss changes with respect to the weights
- finding how the loss changes with respect to the bias
- finding how the loss changes with respect to the input


The parameter gradients (dW and db) are used to update the layer.
The input gradient (dX) is passed backward to earlier layers.[web:32][web:94]


Forward recap:
Let:
- X have shape (B x I)     where B = batch size, I = input dimension
- W have shape (I x O)     where O = output dimension
- b have shape (O)
- Z have shape (B x O)

Then:


  Z = XW + b


In this problem, assume the upstream gradient dL_dZ is already given.
That means some later part of the network has already computed:


  dL_dZ = dL / dZ


and your job is to compute the gradients for this dense layer.[web:16][web:94]


Definitions (batched dense layer backprop):
The standard matrix identities are:[web:17][web:94][web:96]


  dW = X^T * dL_dZ


  db[j] = sum over batch of dL_dZ[k][j]


  dX = dL_dZ * W^T


Elementwise forms:
For each weight gradient:


  dW[i][j] = sum_{k=0..B-1} X[k][i] * dL_dZ[k][j]


For each bias gradient:


  db[j] = sum_{k=0..B-1} dL_dZ[k][j]


For each input gradient:


  dX[k][i] = sum_{j=0..O-1} dL_dZ[k][j] * W[i][j]



Shape rules:
- X has shape (B x I)
- W has shape (I x O)
- dL_dZ has shape (B x O)

Then:
- dW has shape (I x O)
- db has shape (O)
- dX has shape (B x I)

If the shapes are incompatible, return None.


Examples (scalar case):
If:
  X = [[2.0]]
  W = [[3.0]]
  dL_dZ = [[4.0]]

Then:
  dW = [[2.0 * 4.0]] = [[8.0]]
  db = [4.0]
  dX = [[4.0 * 3.0]] = [[12.0]]


Examples (small batch):
If:
  X = [
    [1.0, 2.0],
    [3.0, 4.0],
  ]

  W = [
    [5.0],
    [6.0],
  ]

  dL_dZ = [
    [0.1],
    [0.2],
  ]

Then:

  X^T = [
    [1.0, 3.0],
    [2.0, 4.0],
  ]


  dW = X^T * dL_dZ = [
    [1.0*0.1 + 3.0*0.2],
    [2.0*0.1 + 4.0*0.2],
  ] = [
    [0.7],
    [1.0],
  ]


  db = [0.1 + 0.2] = [0.3]


  W^T = [[5.0, 6.0]]


  dX = dL_dZ * W^T = [
    [0.1*5.0, 0.1*6.0],
    [0.2*5.0, 0.2*6.0],
  ] = [
    [0.5, 0.6],
    [1.0, 1.2],
  ]
*/



struct Solution;



pub struct DenseGrads {
    pub dW: Vec<Vec<f32>>,  // shape: (I x O)
    pub db: Vec<f32>,       // shape: (O)
    pub dX: Vec<Vec<f32>>,  // shape: (B x I)
}



impl Solution {
    pub fn transpose(matrix: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        if matrix.is_empty() {
            return vec![];
        }


        let rows = matrix.len();
        let cols = matrix[0].len();
        let mut out = vec![vec![0.0; rows]; cols];


        for i in 0..rows {
            for j in 0..cols {
                out[j][i] = matrix[i][j];
            }
        }


        out
    }


    pub fn matmul(a: Vec<Vec<f32>>, b: Vec<Vec<f32>>) -> Option<Vec<Vec<f32>>> {
        if a.is_empty() || b.is_empty() {
            return None;
        }


        let a_rows = a.len();
        let a_cols = a[0].len();
        let b_rows = b.len();
        let b_cols = b[0].len();


        for row in &a {
            if row.len() != a_cols {
                return None;
            }
        }


        for row in &b {
            if row.len() != b_cols {
                return None;
            }
        }


        if a_cols != b_rows {
            return None;
        }


        let mut out = vec![vec![0.0; b_cols]; a_rows];


        for i in 0..a_rows {
            for j in 0..b_cols {
                let mut sum = 0.0;
                for k in 0..a_cols {
                    sum += a[i][k] * b[k][j];
                }
                out[i][j] = sum;
            }
        }


        Some(out)
    }


    pub fn dense_backward(
        x: Vec<Vec<f32>>,      // (B x I)
        w: Vec<Vec<f32>>,      // (I x O)
        dL_dZ: Vec<Vec<f32>>,  // (B x O)
    ) -> Option<DenseGrads> {
        // TODO: implement
        //
        // 1. Handle empty cases however you choose, but stay consistent with tests.
        //
        // 2. Check that:
        //    - x is rectangular
        //    - w is rectangular
        //    - dL_dZ is rectangular
        //    - x.len() == dL_dZ.len()             (same batch size B)
        //    - w[0].len() == dL_dZ[0].len()       (same output dimension O)
        //    - x[0].len() == w.len()              (input/output matmul compatibility)
        //
        // 3. Compute:
        //      dW = X^T * dL_dZ
        //
        // 4. Compute:
        //      db[j] = sum over all batch rows of dL_dZ[k][j]
        //
        // 5. Compute:
        //      dX = dL_dZ * W^T
        //
        // 6. Return:
        //      Some(DenseGrads { dW, db, dX })

        if x.is_empty() || w.is_empty() || dL_dZ.is_empty() {
            return None;
        }

        let x_cols = x[0].len();
        let w_cols = w[0].len();
        let dz_cols = dL_dZ[0].len();

        for row in &x {
            if row.len() != x_cols {
                return None;
            }
        }

        for row in &w {
            if row.len() != w_cols {
                return None;
            }
        }

        for row in &dL_dZ {
            if row.len() != dz_cols {
                return None;
            }
        }

        if x.len() != dL_dZ.len() {
            return None;
        }

        if w_cols != dz_cols {
            return None;
        }

        if x_cols != w.len() {
            return None;
        }

        let dW = Self::matmul(Self::transpose(x), dL_dZ);
        let dX = Self::matmul(dL_dZ, Self::transpose(w));

        let rows = dL_dZ.len();
        let cols = dL_dZ[0].len();
        let mut db = Vec::new();

        for i in 0..rows {
            for j in 0..cols {
                db[i] += dL_dZ[i][j];
            }
        }

        Some(DenseGrads { dW, db, dX} )
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
    expected: Option<DenseGrads>,
) {
    let actual = Solution::dense_backward(x.clone(), w.clone(), dL_dZ.clone());


    let passed = match (&actual, &expected) {
        (Some(a), Some(e)) => {
            matrices_close(&a.dW, &e.dW, 1e-4)
                && vectors_close(&a.db, &e.db, 1e-4)
                && matrices_close(&a.dX, &e.dX, 1e-4)
        }
        (None, None) => true,
        _ => false,
    };


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


    match &expected {
        Some(g) => {
            println!("{}Expected dW:{} {:?}", yellow, reset, g.dW);
            println!("{}Expected db:{} {:?}", yellow, reset, g.db);
            println!("{}Expected dX:{} {:?}", yellow, reset, g.dX);
        }
        None => {
            println!("{}Expected:{} None", yellow, reset);
        }
    }


    match &actual {
        Some(g) => {
            println!("{}Actual dW:{}   {:?}", yellow, reset, g.dW);
            println!("{}Actual db:{}   {:?}", yellow, reset, g.db);
            println!("{}Actual dX:{}   {:?}", yellow, reset, g.dX);
        }
        None => {
            println!("{}Actual:{}   None", yellow, reset);
        }
    }


    if passed {
        println!("{}✅ PASS{}", green, reset);
    } else {
        println!("{}❌ FAIL{}", red, reset);
    }
}


fn main() {
    println!("\x1b[1m\x1b[35mNN Runner: Dense Layer Backward\x1b[0m");


    print_result(
        "Test Case 1: scalar dense layer",
        vec![vec![2.0]],
        vec![vec![3.0]],
        vec![vec![4.0]],
        Some(DenseGrads {
            dW: vec![vec![8.0]],
            db: vec![4.0],
            dX: vec![vec![12.0]],
        }),
    );


    print_result(
        "Test Case 2: small batched dense layer",
        vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        vec![vec![5.0], vec![6.0]],
        vec![vec![0.1], vec![0.2]],
        Some(DenseGrads {
            dW: vec![vec![0.7], vec![1.0]],
            db: vec![0.3],
            dX: vec![vec![0.5, 0.6], vec![1.0, 1.2]],
        }),
    );


    print_result(
        "Test Case 3: zero upstream gradient",
        vec![vec![1.0, -2.0], vec![3.0, 4.0]],
        vec![vec![0.5, -1.0], vec![2.0, 3.0]],
        vec![vec![0.0, 0.0], vec![0.0, 0.0]],
        Some(DenseGrads {
            dW: vec![vec![0.0, 0.0], vec![0.0, 0.0]],
            db: vec![0.0, 0.0],
            dX: vec![vec![0.0, 0.0], vec![0.0, 0.0]],
        }),
    );


    print_result(
        "Test Case 4: shape mismatch",
        vec![vec![1.0, 2.0]],
        vec![vec![1.0]],
        vec![vec![0.5, 0.6]],
        None,
    );
}
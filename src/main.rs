/*
NN Step 14. SGD Parameter Update
[src/solutions/nn/14-SGD.rs]


Goal:
Implement one Stochastic Gradient Descent (SGD) update step
for a dense layer's weights and bias.


Why this matters for neural networks:
Backpropagation tells us how the loss changes with respect to
the parameters:
- dW = gradient of loss w.r.t. W
- db = gradient of loss w.r.t. b

But gradients alone do not train the model.
To actually learn, we must update the parameters in the direction
that reduces the loss.

SGD performs the basic update:[web:87][web:108]

  parameter = parameter - learning_rate * gradient

For a dense layer, that means:[web:87][web:109]

  W[i][j] = W[i][j] - lr * dW[i][j]
  b[j]    = b[j]    - lr * db[j]


Definitions:
Let:
- W have shape (I x O)
- b have shape (O)
- dW have shape (I x O)
- db have shape (O)
- lr be a positive scalar learning rate

Then the updated parameters are:

  new_W[i][j] = W[i][j] - lr * dW[i][j]
  new_b[j]    = b[j]    - lr * db[j]


Shape rules:
- W and dW must have the same shape
- b and db must have the same length
- W and dW must both be rectangular
- If shapes are incompatible, return None


Examples (scalar case):
If:
  W  = [[3.0]]
  b  = [1.0]
  dW = [[8.0]]
  db = [4.0]
  lr = 0.1

Then:

  new_W = [[3.0 - 0.1*8.0]] = [[2.2]]
  new_b = [1.0 - 0.1*4.0]   = [0.6]


Examples (small matrix):
If:
  W = [
    [1.0, 2.0],
    [3.0, 4.0],
  ]

  b = [0.5, -0.5]

  dW = [
    [0.1, 0.2],
    [0.3, 0.4],
  ]

  db = [0.5, -1.0]

  lr = 0.1

Then:

  new_W = [
    [1.0 - 0.1*0.1, 2.0 - 0.1*0.2],
    [3.0 - 0.1*0.3, 4.0 - 0.1*0.4],
  ] = [
    [0.99, 1.98],
    [2.97, 3.96],
  ]

  new_b = [
    0.5 - 0.1*0.5,
   -0.5 - 0.1*(-1.0),
  ] = [
    0.45,
   -0.4,
  ]


Notes:
- This problem is just the raw SGD update step.
- No momentum, Adam, RMSProp, or weight decay here.
- Assume lr is valid for this exercise.
*/



struct Solution;

pub struct Params {
    pub w: Vec<Vec<f32>>,
    pub b: Vec<f32>,
}

impl Solution {
    pub fn sgd_update(
        w: Vec<Vec<f32>>,
        b: Vec<f32>,
        dW: Vec<Vec<f32>>,
        db: Vec<f32>,
        lr: f32,
    ) -> Option<Params> {
        // TODO: implement
        //
        // 1. Return None if:
        //    - w, b, dW, or db are empty
        //    - w or dW are non-rectangular
        //    - w and dW do not have the same shape
        //    - b.len() != db.len()
        //
        // 2. Compute:
        //      new_w[i][j] = w[i][j] - lr * dW[i][j]
        //
        // 3. Compute:
        //      new_b[j] = b[j] - lr * db[j]
        //
        // 4. Return:
        //      Some(Params { w: new_w, b: new_b })

        unimplemented!()
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
    w: Vec<Vec<f32>>,
    b: Vec<f32>,
    dW: Vec<Vec<f32>>,
    db: Vec<f32>,
    lr: f32,
    expected: Option<Params>,
) {
    let actual = Solution::sgd_update(w.clone(), b.clone(), dW.clone(), db.clone(), lr);

    let passed = match (&actual, &expected) {
        (Some(a), Some(e)) => {
            matrices_close(&a.w, &e.w, 1e-4)
                && vectors_close(&a.b, &e.b, 1e-4)
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
    println!("{}W:{}   {:?}", yellow, reset, w);
    println!("{}b:{}   {:?}", yellow, reset, b);
    println!("{}dW:{}  {:?}", yellow, reset, dW);
    println!("{}db:{}  {:?}", yellow, reset, db);
    println!("{}lr:{}  {:?}", yellow, reset, lr);

    match &expected {
        Some(p) => {
            println!("{}Expected W:{} {:?}", yellow, reset, p.w);
            println!("{}Expected b:{} {:?}", yellow, reset, p.b);
        }
        None => {
            println!("{}Expected:{} None", yellow, reset);
        }
    }

    match &actual {
        Some(p) => {
            println!("{}Actual W:{}   {:?}", yellow, reset, p.w);
            println!("{}Actual b:{}   {:?}", yellow, reset, p.b);
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
    println!("\x1b[1m\x1b[35mNN Runner: SGD Update\x1b[0m");

    print_result(
        "Test Case 1: scalar update",
        vec![vec![3.0]],
        vec![1.0],
        vec![vec![8.0]],
        vec![4.0],
        0.1,
        Some(Params {
            w: vec![vec![2.2]],
            b: vec![0.6],
        }),
    );

    print_result(
        "Test Case 2: small matrix update",
        vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        vec![0.5, -0.5],
        vec![vec![0.1, 0.2], vec![0.3, 0.4]],
        vec![0.5, -1.0],
        0.1,
        Some(Params {
            w: vec![vec![0.99, 1.98], vec![2.97, 3.96]],
            b: vec![0.45, -0.4],
        }),
    );

    print_result(
        "Test Case 3: zero gradients",
        vec![vec![1.5, -2.0], vec![0.0, 3.0]],
        vec![0.2, -0.1],
        vec![vec![0.0, 0.0], vec![0.0, 0.0]],
        vec![0.0, 0.0],
        0.01,
        Some(Params {
            w: vec![vec![1.5, -2.0], vec![0.0, 3.0]],
            b: vec![0.2, -0.1],
        }),
    );

    print_result(
        "Test Case 4: shape mismatch",
        vec![vec![1.0, 2.0]],
        vec![0.5],
        vec![vec![0.1], vec![0.2]],
        vec![0.3],
        0.1,
        None,
    );
}
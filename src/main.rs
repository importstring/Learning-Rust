/*
NN Step 12. MSE Backward
[src/solutions/nn/12-MSE-Backward.rs]


Goal:
Given predictions and targets for a batch, compute the gradient
of Mean Squared Error (MSE) with respect to the predictions.


Why this matters for neural networks:
To train a network, you need the gradient of the loss with respect
to the model outputs. For MSE, this gradient is simple and fully
elementwise, which makes it a great first loss to backpropagate.

Once you can get dL/dPred for MSE, you can feed that into your
ReLU backward and dense backward steps to propagate gradients
all the way to the weights.


Definition (scalar, single sample):
For a single prediction y_hat and target y:

  L = (1/2) * (y_hat - y)^2

Then:

  dL/dy_hat = (y_hat - y)

(The 1/2 is there so the 2 from the derivative cancels out.)


Definition (vector, single sample):
For a prediction vector y_hat[j] and target y[j], j = 0..O-1:

  L = (1/2) * sum_{j=0..O-1} (y_hat[j] - y[j])^2

Then:

  dL/dy_hat[j] = (y_hat[j] - y[j])


Definition (batched, no averaging over batch):
For a batch of B samples, each of length O:

  L_batch = (1/2) * sum_{i=0..B-1} sum_{j=0..O-1} (pred[i][j] - target[i][j])^2

Then for each element:

  dL/dPred[i][j] = (pred[i][j] - target[i][j])

(In this exercise, we do NOT divide by B or O; the scale factor
is left as 1 for simplicity. You can add averaging later.)


Shape rules:
- pred has shape (B x O)
- target has shape (B x O)
- dL_dPred has shape (B x O)
- if shapes do not match, return None


Examples (single sample, 1D):

pred = [[3.0, 5.0]]
target = [[1.0, 2.0]]

Differences = [[2.0, 3.0]]

dL_dPred = [[2.0, 3.0]]


Examples (batch):

pred = [
  [1.0, 2.0],
  [3.0, 4.0],
]

target = [
  [0.0, 0.0],
  [1.0, 1.0],
]

differences = [
  [1.0, 2.0],
  [2.0, 3.0],
]

dL_dPred = differences (same values, same shape)
*/



struct Solution;



impl Solution {
    pub fn mse_backward(
        pred: Vec<Vec<f32>>,
        target: Vec<Vec<f32>>,
    ) -> Option<Vec<Vec<f32>>> {
        // TODO: implement
        //
        // 1. Check shapes: same number of rows and same number of cols.
        //    If not, return None.
        //
        // 2. Allocate an output matrix dL_dPred with the same shape as pred.
        //
        // 3. For each (i, j):
        //        dL_dPred[i][j] = pred[i][j] - target[i][j]
        //
        //    We are using the "no averaging" version here:
        //        L = (1/2) * sum (pred - target)^2
        //    so dL/dPred = (pred - target).
        //
        // 4. Return Some(dL_dPred).

        if pred.is_empty() || target.is_empty() {
            return Some(Vec::new());
        }

        if pred[0].len() != target[0].len() {
            return None;
        }

        let rows = pred.len();
        let cols = pred[0].len();
        let mut out = vec![vec![0.0; cols]; rows];

        for i in 0..rows {
            for j in 0..cols {
                out[i][j] = pred[i][j] - target[i][j];
            }
        }

        Some(out)
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



fn print_result(
    case_name: &str,
    pred: Vec<Vec<f32>>,
    target: Vec<Vec<f32>>,
    expected: Option<Vec<Vec<f32>>>,
) {
    let actual = Solution::mse_backward(pred.clone(), target.clone());


    let passed = match (&actual, &expected) {
        (Some(a), Some(e)) => matrices_close(a, e, 1e-5),
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
    println!("{}Pred:{}    {:?}", yellow, reset, pred);
    println!("{}Target:{}  {:?}", yellow, reset, target);
    println!("{}Expected:{} {:?}", yellow, reset, expected);
    println!("{}Actual:{}   {:?}", yellow, reset, actual);


    if passed {
        println!("{}✅ PASS{}", green, reset);
    } else {
        println!("{}❌ FAIL{}", red, reset);
    }
}



fn main() {
    println!("\x1b[1m\x1b[35mNN Runner: MSE Backward\x1b[0m");


    // Test 1: single sample, 1D
    // pred = [[3.0, 5.0]]
    // target = [[1.0, 2.0]]
    // dL_dPred = [[2.0, 3.0]]
    print_result(
        "Test Case 1: single sample 1D",
        vec![vec![3.0, 5.0]],
        vec![vec![1.0, 2.0]],
        Some(vec![vec![2.0, 3.0]]),
    );


    // Test 2: small batch, 2D
    print_result(
        "Test Case 2: small batch",
        vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        vec![vec![0.0, 0.0], vec![1.0, 1.0]],
        Some(vec![vec![1.0, 2.0], vec![2.0, 3.0]]),
    );


    // Test 3: zeros → gradient is just (0 - target)
    print_result(
        "Test Case 3: zero predictions",
        vec![vec![0.0, 0.0]],
        vec![vec![1.5, -2.5]],
        Some(vec![vec![-1.5, 2.5]]),
    );


    // Test 4: shape mismatch → None
    print_result(
        "Test Case 4: shape mismatch",
        vec![vec![1.0, 2.0]],
        vec![vec![1.0]],
        None,
    );


    // Test 5: empty inputs → treat as valid zero-size gradient
    print_result(
        "Test Case 5: empty",
        vec![],
        vec![],
        Some(vec![]),
    );
}
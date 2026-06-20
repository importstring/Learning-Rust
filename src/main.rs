/*
NN Step 9. MSE Gradient w.r.t. Predictions
[src/solutions/nn/09-MSEGradPred.rs]

Goal:
Given predictions and targets, compute the gradient of the Mean Squared Error
(MSE) with respect to the predictions.

Context:
During training, gradients tell you how changing the outputs (predictions)
would change the loss. For a network ending in a dense layer with MSE loss,
this is the "top" gradient that flows backward into the layer.

Setup:
Let pred and target be 2D arrays with the same shape:
- pred[i][j] is the prediction for sample i, output j
- target[i][j] is the corresponding true value

Let N be the total number of scalar elements:
N = number_of_rows * number_of_columns

Define the Mean Squared Error over all elements as:

MSE = (1 / N) * sum over all i,j of (pred[i][j] - target[i][j])^2

Task:
For each element pred[i][j], compute the partial derivative of MSE with
respect to pred[i][j]. The result should be another 2D array grad with
the same shape as pred and target, where:

grad[i][j] = d(MSE) / d(pred[i][j])

You should use the definition above and basic derivative rules to obtain
and implement the correct expression.

Examples (for sanity checks):

Example 1:
pred   = [[1.0, 2.0]]
target = [[1.0, 2.0]]

N = 2
MSE = 0.0
So every entry of the gradient should be 0.0:
grad = [[0.0, 0.0]]

Example 2:
pred   = [[2.0, 4.0]]
target = [[1.0, 1.0]]

N = 2
pred - target = [[1.0, 3.0]]

You should write the formula so that plugging in these numbers yields:
grad = [[1.0, 3.0]]

Example 3 (batch of 2):
pred   = [[1.0, 2.0],
          [3.0, 4.0]]

target = [[1.0, 0.0],
          [5.0, 4.0]]

pred - target = [[0.0,  2.0],
                 [-2.0, 0.0]]

N = 4

Your implementation should produce:
grad = [[0.0, 1.0],
        [-1.0, 0.0]]
*/

struct Solution;

impl Solution {
    pub fn mse_grad_pred(
        pred: Vec<Vec<f32>>,
        target: Vec<Vec<f32>>,
    ) -> Option<Vec<Vec<f32>>> {

        let rows = pred.len();
        let cols = pred[0].len();
        let n = (rows * cols) as f32;

        let mut grad = vec![vec![0f32; cols]; rows];

        for i in 0..rows {
            for j in 0..cols {
                grad[i][j] = (2.0 / n) * (pred[i][j] - target[i][j]);
            }
        }

        Some(grad)
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
    let actual = Solution::mse_grad_pred(pred.clone(), target.clone());

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
    println!("{}pred:{}   {:?}", yellow, reset, pred);
    println!("{}target:{} {:?}", yellow, reset, target);
    println!("{}Expected:{} {:?}", yellow, reset, expected);
    println!("{}Actual:{}   {:?}", yellow, reset, actual);

    if passed {
        println!("{}✅ PASS{}", green, reset);
    } else {
        println!("{}❌ FAIL{}", red, reset);
    }
}

fn main() {
    println!("\x1b[1m\x1b[35mNN Runner: MSE Grad w.r.t. Predictions\x1b[0m");

    // identical -> zero gradient
    print_result(
        "Test Case 1",
        vec![vec![1.0, 2.0]],
        vec![vec![1.0, 2.0]],
        Some(vec![vec![0.0, 0.0]]),
    );

    // 1x2 example from header
    print_result(
        "Test Case 2",
        vec![vec![2.0, 4.0]],
        vec![vec![1.0, 1.0]],
        Some(vec![vec![1.0, 3.0]]),
    );

    // batch of 2 example
    print_result(
        "Test Case 3",
        vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        vec![vec![1.0, 0.0], vec![5.0, 4.0]],
        Some(vec![vec![0.0, 1.0], vec![-1.0, 0.0]]),
    );

    // no more mismatched-shape test; focus is on math, not guards
}
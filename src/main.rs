/*
NN Step 10. Sigmoid Activation
[src/solutions/nn/10-Sigmoid.rs]


Goal:
Apply the sigmoid function element-wise to a 2D matrix.


Why this matters for neural networks:
Sigmoid squashes any real number into (0, 1), making it useful for:
- Output layers in binary classification
- Gating mechanisms (LSTMs, attention)

Unlike ReLU, sigmoid is differentiable everywhere, and its derivative
has a clean closed form that you'll use directly in backprop:

  sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))


Definition:
Given a matrix M with shape (rows x cols):

  sigmoid(x) = 1 / (1 + e^(-x))

  out[i][j] = 1.0 / (1.0 + (-M[i][j]).exp())


Shape rules:
- Output shape is identical to input shape.
- Empty matrix returns empty matrix.


Numerical note:
For large negative x, e^(-x) overflows. Rust's f32::exp() handles
this gracefully (returns inf, so sigmoid → 0), but be aware.


Examples:


M = [
  [0.0, 1.0],
  [-1.0, 2.0],
]

out ≈ [
  [0.5,     0.7311],
  [0.2689,  0.8808],
]


M = [[0.0]]

out = [[0.5]]


M = []

out = []
*/


struct Solution;


impl Solution {
    pub fn sigmoid(matrix: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        // your solution here
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
    matrix: Vec<Vec<f32>>,
    expected: Vec<Vec<f32>>,
) {
    let actual = Solution::sigmoid(matrix.clone());

    let passed = matrices_close(&actual, &expected, 1e-4);

    let green = "\x1b[32m";
    let red = "\x1b[31m";
    let cyan = "\x1b[36m";
    let yellow = "\x1b[33m";
    let bold = "\x1b[1m";
    let reset = "\x1b[0m";

    println!("\n{}{}{}{}", bold, cyan, case_name, reset);
    println!("{}Matrix:{} {:?}", yellow, reset, matrix);
    println!("{}Expected:{} {:?}", yellow, reset, expected);
    println!("{}Actual:{}   {:?}", yellow, reset, actual);

    if passed {
        println!("{}✅ PASS{}", green, reset);
    } else {
        println!("{}❌ FAIL{}", red, reset);
    }
}


fn main() {
    println!("\x1b[1m\x1b[35mNN Runner: Sigmoid\x1b[0m");

    print_result(
        "Test Case 1: standard values",
        vec![vec![0.0, 1.0], vec![-1.0, 2.0]],
        vec![
            vec![0.5, 0.7310586],
            vec![0.26894143, 0.880797],
        ],
    );

    print_result(
        "Test Case 2: zero",
        vec![vec![0.0]],
        vec![vec![0.5]],
    );

    print_result(
        "Test Case 3: large positive (→ 1)",
        vec![vec![100.0]],
        vec![vec![1.0]],
    );

    print_result(
        "Test Case 4: large negative (→ 0)",
        vec![vec![-100.0]],
        vec![vec![0.0]],
    );

    print_result(
        "Test Case 5: empty matrix",
        vec![],
        vec![],
    );

    print_result(
        "Test Case 6: multi-row",
        vec![
            vec![-2.0, -1.0, 0.0],
            vec![1.0, 2.0, 3.0],
        ],
        vec![
            vec![0.11920292, 0.26894143, 0.5],
            vec![0.7310586, 0.880797, 0.95257413],
        ],
    );
}
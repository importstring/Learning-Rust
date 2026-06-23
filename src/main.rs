/*
NN Step 11. Softmax Activation
[src/solutions/nn/11-Softmax.rs]


Goal:
Apply the softmax function row-wise to a 2D matrix.


Why this matters for neural networks:
Softmax turns arbitrary real-valued scores (logits) into a probability
distribution over classes.[web:12][web:16]
Each row is typically one sample’s class scores; after softmax:
- All entries are in (0, 1)
- Each row sums to 1
- Larger scores become larger probabilities


Definition (row-wise softmax):
Given a matrix M with shape (rows x cols), for each row i:

  let z = M[i]
  softmax(z)_j = exp(z_j) / sum_k exp(z_k)

So:

  out[i][j] = exp(M[i][j]) / sum_{k=0..cols-1} exp(M[i][k])


Numerical stability:
Directly exponentiating large values can overflow.
A standard trick is to subtract the row max before exponentiating:[web:13][web:16]

  let max_z = max_j M[i][j]
  let exps_j = exp(M[i][j] - max_z)
  let denom = sum_j exps_j
  out[i][j] = exps_j / denom

This keeps values in a safer numeric range while leaving
the result unchanged mathematically.


Shape rules:
- Input shape: (rows x cols)
- Output shape: (rows x cols)
- If the matrix is empty (rows == 0), return an empty matrix.
- If a row is empty (cols == 0), return an empty row for that row.


Examples:

M = [
  [1.0, 2.0, 3.0],
]

softmax(M[0]) ≈ [
  0.0900, 0.2447, 0.6652
]


M = [
  [1.0, 2.0, 3.0],
  [1.0, 2.0, 3.0],
]

out has two identical rows, each ≈ [0.0900, 0.2447, 0.6652]


M = []

out = []
*/


struct Solution;


impl Solution {
    pub fn softmax(matrix: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        
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


// Helper to check that each row sums to ~1.0
fn rows_sum_to_one(matrix: &[Vec<f32>], eps: f32) -> bool {
    for row in matrix {
        let sum: f32 = row.iter().sum();
        if (sum - 1.0).abs() > eps {
            return false;
        }
    }
    true
}


fn print_result(
    case_name: &str,
    matrix: Vec<Vec<f32>>,
    expected: Vec<Vec<f32>>,
) {
    let actual = Solution::softmax(matrix.clone());

    let values_ok = matrices_close(&actual, &expected, 1e-4);
    let sums_ok = rows_sum_to_one(&actual, 1e-4);
    let passed = values_ok && sums_ok;

    let green = "\x1b[32m";
    let red = "\x1b[31m";
    let cyan = "\x1b[36m";
    let yellow = "\x1b[33m";
    let bold = "\x1b[1m";
    let reset = "\x1b[0m";

    println!("\n{}{}{}{}", bold, cyan, case_name, reset);
    println!("{}Matrix:{}   {:?}", yellow, reset, matrix);
    println!("{}Expected:{} {:?}", yellow, reset, expected);
    println!("{}Actual:{}   {:?}", yellow, reset, actual);
    println!("Row sums ≈ 1.0: {}", sums_ok);

    if passed {
        println!("{}✅ PASS{}", green, reset);
    } else {
        println!("{}❌ FAIL{}", red, reset);
    }
}


fn main() {
    println!("\x1b[1m\x1b[35mNN Runner: Softmax\x1b[0m");

    print_result(
        "Test Case 1: single row",
        vec![vec![1.0, 2.0, 3.0]],
        vec![vec![
            0.09003057,
            0.24472848,
            0.66524094,
        ]],
    );

    print_result(
        "Test Case 2: two identical rows",
        vec![
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
        ],
        vec![
            vec![0.09003057, 0.24472848, 0.66524094],
            vec![0.09003057, 0.24472848, 0.66524094],
        ],
    );

    print_result(
        "Test Case 3: negative values",
        vec![vec![-1.0, 0.0, 1.0]],
        vec![vec![
            0.09003057,
            0.24472848,
            0.66524094,
        ]],
    );

    print_result(
        "Test Case 4: larger spread",
        vec![vec![0.0, 5.0]],
        vec![vec![
            0.00669285,
            0.99330715,
        ]],
    );

    print_result(
        "Test Case 5: empty matrix",
        vec![],
        vec![],
    );

    print_result(
        "Test Case 6: empty row",
        vec![vec![]],
        vec![vec![]],
    );
}
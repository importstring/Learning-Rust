/*
NN Step 11. ReLU Backward
[src/solutions/nn/11-ReLUBackward.rs]


Goal:
Implement the backward pass for ReLU activation on a 2D matrix.


Why this matters for neural networks:
In a forward pass, ReLU is applied elementwise:


out[i][j] = max(0, input[i][j])


During backpropagation, we are given an upstream gradient matrix
dL_dOut with the same shape as the ReLU output.
We want the gradient of the loss with respect to the original input:


dL_dInput


ReLU only passes gradient through entries where the original input
was strictly greater than 0.
If the input was less than or equal to 0, the gradient becomes 0.


Definition:
For each element:


if input[i][j] > 0:
    dL_dInput[i][j] = dL_dOut[i][j]
else:
    dL_dInput[i][j] = 0.0


Shape rules:
- input has shape (rows x cols)
- dL_dOut must have the same shape
- result dL_dInput has shape (rows x cols)
- if shapes do not match, return None


Examples:


input = [
  [-1.0,  0.0,  2.0],
  [ 3.0, -4.0,  5.0],
]


dL_dOut = [
  [10.0, 20.0, 30.0],
  [40.0, 50.0, 60.0],
]


dL_dInput = [
  [ 0.0,  0.0, 30.0],
  [40.0,  0.0, 60.0],
]


input = [
  [1.5, 2.5],
]


dL_dOut = [
  [0.1, -0.2],
]


dL_dInput = [
  [0.1, -0.2],
]
*/


struct Solution;


impl Solution {
    pub fn relu_backward(
        input: Vec<Vec<f32>>,
        dL_dOut: Vec<Vec<f32>>,
    ) -> Option<Vec<Vec<f32>>> {

        if input.is_empty() {
            return None;
        }

        let rows = input.len();
        let cols = if rows > 0 { input[0].len() } else { 0 };
        let mut out = vec![vec![0.0; cols]; rows];
        
        for i in 0..rows {
            for j in 0..cols {
                if input[i][j] > 0.0 {
                    out[i][j] = dL_dOut[i][j];
                }
                else {
                    out[i][j] = 0.0;
                }
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
    input: Vec<Vec<f32>>,
    dL_dOut: Vec<Vec<f32>>,
    expected: Option<Vec<Vec<f32>>>,
) {
    let actual = Solution::relu_backward(input.clone(), dL_dOut.clone());


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
    println!("{}Input:{}   {:?}", yellow, reset, input);
    println!("{}dL_dOut:{} {:?}", yellow, reset, dL_dOut);
    println!("{}Expected:{} {:?}", yellow, reset, expected);
    println!("{}Actual:{}   {:?}", yellow, reset, actual);


    if passed {
        println!("{}✅ PASS{}", green, reset);
    } else {
        println!("{}❌ FAIL{}", red, reset);
    }
}


fn main() {
    println!("\x1b[1m\x1b[35mNN Runner: ReLU Backward\x1b[0m");


    print_result(
        "Test Case 1",
        vec![vec![-1.0, 0.0, 2.0], vec![3.0, -4.0, 5.0]],
        vec![vec![10.0, 20.0, 30.0], vec![40.0, 50.0, 60.0]],
        Some(vec![vec![0.0, 0.0, 30.0], vec![40.0, 0.0, 60.0]]),
    );


    print_result(
        "Test Case 2",
        vec![vec![1.5, 2.5]],
        vec![vec![0.1, -0.2]],
        Some(vec![vec![0.1, -0.2]]),
    );


    print_result(
        "Test Case 3",
        vec![vec![0.0, -2.0]],
        vec![vec![7.0, 8.0]],
        Some(vec![vec![0.0, 0.0]]),
    );


    print_result(
        "Test Case 4",
        vec![vec![1.0, 2.0]],
        vec![vec![3.0]],
        None,
    );


    print_result(
        "Test Case 5",
        vec![],
        vec![],
        Some(vec![]),
    );
}
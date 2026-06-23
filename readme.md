## Learning AI without Python Libraries

So I've been doing ML projects for a while now but I've realized I've just been calling libraries and calling it machine learning. Sure it works but I'm not understanding deeply what's going on. That's why I'm learning Rust. It will force me to learn algorithms, in-depth how AI works, and fill my curiousity for the language.

My final challenge will be building a neural network and I'll be working up to it by building mini projects and solving easy–hard leatcode problems.

The plan is to turn this into this very helpful environment with hints and syntax suggestions and links to Kaggle and Leatcode to make learning Rust more fun.

I'm working in `src/solutions/nn` right now! 

Follow along and star!
--> can find the dev journal in root


### 📌 Today's Pinned Entry 
<h4 style="margin-bottom:0;">First Weeks into Learning Rust</h4>
<p style="margin-top:0; font-size:0.85em;">
  <em>2026-06-22 · Rust Learning week 3</em>
</p>

I've been working on MSE, matrix multiplication, dot products, relu, and then working that together to create a super simply feed forward network.

Basically I'm building myself up from the basics so that I have a strong understanding of how everything works.

It worked super nicely because I'd build in the bias function, the ReLu, and dot products and then on the very next mini challenge, use my old code for the new problem.

```Rust
pub fn matmul(...) { ...}
pub fn add_bias(...) {...}
pub fn relu(...) {...}

pub fn dense_forward(
    x: Vec<Vec<f32>>,
    w: Vec<Vec<f32>>,
    b: Vec<f32>,
) -> Option<Vec<Vec<f32>>> {
    if x.is_empty() || w.is_empty() || b.is_empty() {
        return None;
    }
    
    let z = Self::matmul(x, w);
    let z_b = Self::add_bias(z?, b);
    let y = Self::relu(z_b?);
    Some(y)
```

How I've been learning thusfar:
I've been leaning on AI to roadmap my progression from skillset to skillset and have it generate LeatCode style files for me to build ML stuff. It's been working great but I'm hoping I can reach out to someone more senior for some guidance on how they'd go from here.

```Rust
fn main() {
  println!("\x1b[1m\x1b[35mNN Runner: Dense Layer Forward\x1b[0m");

  // Identity-like layer, no bias
  print_result(
      "Test Case 1",
      vec![vec![1.0, 2.0]],
      vec![vec![1.0, 0.0], vec![0.0, 1.0]],
      vec![0.0, 0.0],
      Some(vec![vec![1.0, 2.0]]),
  );

  // ...    
}
```

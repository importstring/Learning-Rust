## First Hours into Learning Rust [Thu Jun 11]

Lot's of confusion but my focus has been to avoid LLM code like it's the plague. Autocorrect is off and so is Copilot.

I'm reading through the amazing Rust documentation along with doing Leatcode problems.

As for the use of AI, I've been using Perplexity.ai for choosing Leatcode problems and writing tests for them exlcusively.

All code is human written. I've loved the look of Rust code and I always imitate it in Python anyways so I'm hoping this keeps up. It's a little bit of a learning curve.
As long as I keep challenging myself and going through the docs I'll eventually learn.

The syntax too is starting to make sense but I have to think about it.

```Rust
Vec<i32> // Right cos Vector = Array = List and i32 interger 32 bits
for i in 0..nums.len() // 0..nums like the ... in math and .len() is an improvement
-> Vec<i32> // is also phenominal because no typing library is required
return // Early return and auto returning is amazing
```

Still wrapping my head around som errors with pointers and borrowing. Anyways did two Leatcode problems in rust.

```Rust
impl Solution {
    pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
        for i in 0..nums.len() {
            for j in (i+1)..nums.len() {
                if nums[i] + nums[j] == target {
                    return vec![i as i32, j as i32];
                }
            }
        }
    }
}

// Which I optimized using hashmaps to
use std::collections::HashMap;
let mut map: HashMap<i32, usize> = HashMap::new();

for (i, &val) in nums.iter().enumerate() {
    let mut complement = target - val;
    if let Some(&j) = map.get(&complement) {
        return vec![j as i32, i as i32];
    }
    else {
        map.insert(val, i);
    }
}


// &

impl Solution {
    pub fn is_palindrome(s: String) -> bool {
        let filtered: String = s
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect();

        let reversed: String = filtered
            .chars()
            .rev()
            .collect();

        filtered == reversed
    }
}
```

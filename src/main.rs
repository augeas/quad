
use ndarray::prelude::*;
use ndarray::Array;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

fn rand_coeffs(n: usize) -> (Vec<String>, Array<f32, Ix2>, Array<f32, Ix3>) {
	let i = Array::random((n, 12), Uniform::new_inclusive(0, 24).unwrap());
	let c = (0.1 * (i.mapv(|elem| elem as f32) - 12.0)).into_shape_with_order((n, 2, 6)).unwrap();
	let bias = c.slice(s![.., .., 0]).to_owned();
	let coeff = c.slice(s![.., .., 1..]).to_owned();
	let chars = (i + 65).mapv(|chr| chr as u8 as char);
	let names: Vec<String> = chars.outer_iter().map(String::from_iter).collect();
	println!("{:?}", names);
	return (names, bias, coeff);
}

fn main() {
	rand_coeffs(4);
	println!("Hello, world!");
}

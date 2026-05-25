
use ndarray::prelude::*;
use ndarray::Array;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

fn rand_coeffs(n: usize) -> (Vec<String>, Array<f32, Ix3>, Array<f32, Ix3>) {
	let i = Array::random((n, 12), Uniform::new_inclusive(0, 24).unwrap());
	let c = (0.1 * (i.mapv(|elem| elem as f32) - 12.0)).into_shape_with_order((n, 2, 6)).unwrap();
	let bias = c.slice(s![.., .., 0..1]).to_owned();
	let coeff = c.slice(s![.., .., 1..]).to_owned();
	let chars = (i + 65).mapv(|chr| chr as u8 as char);
	let names: Vec<String> = chars.outer_iter().map(String::from_iter).collect();
	return (names, bias, coeff);
}

fn init_var(n: usize) -> Array<f32, Ix3> {
	return Array3::<f32>::ones((n, 2, 1)) * 0.01;
}

fn var_vec(var: &Array<f32, Ix3>) -> Array<f32, Ix3> {
	let n = var.shape()[0];
	let mut vec = Array3::<f32>::zeros((n, 5, 1));
	
	for i in 0..n {
		let v = var.slice(s![i, .., ..]);
		let v_sq = v.dot(&v.t()).into_shape_with_order((4, 1)).unwrap();
		vec.slice_mut(s![i, 1.., ..]).assign(&v_sq);
	}
	vec.slice_mut(s![.., 0, ..]).assign(&var.slice(s![.., 0, ..]));
	vec.slice_mut(s![.., 3, ..]).assign(&var.slice(s![.., 1, ..]));

	return vec
}

fn quad_iterate(var: &Array<f32, Ix3>, bias: &Array<f32, Ix3>, coeff: &Array<f32, Ix3>) -> Array<f32, Ix3> {
	let n = var.shape()[0];
	let vec = var_vec(&var);
	let mut new_var = Array3::<f32>::zeros((n, 2, 1));
	
	for i in 0..n {
		let point = coeff.slice(s![i, .., ..]).dot(&(vec.slice(s![i, .., ..])));
		new_var.slice_mut(s![i, .., ..]).assign(&point);
	}
	return bias + &new_var;
}

fn main() {
	let (names, bias, coeffs) = rand_coeffs(4);
	let v = init_var(4);
	//let vv = var_vec(&v);
	println!("{:?}", v);
	println!("{:?}", quad_iterate(&v, &bias, &coeffs));
	//println!("Hello, world!");
}



mod image;

use anyhow::{Context, Result};
use opencv::{
    boxed_ref::BoxedRef,
    core::{
        Mat, MatTrait, MatTraitConst, MatTraitConstManual, MatTraitManual, Point2i, Rect, Scalar, Vec3b,
        Vector,
    },
};
use rand::prelude::RngExt;

fn main() -> Result<()> {
    let params = Vector::default();
    core(&params)?;
    Ok(())
}

fn core(params: &Vector<i32>) -> Result<()> {
    let patch = 7;

    let im_src = image::read("Source.jpg")?;
    let im_target = image::read("Target.jpg")?;
    let src_border = image::border(&im_src, patch)?;
    let target_border = image::border(&im_target, patch)?;

    let nnf = initialize_nnf(&im_src, &im_target).context("Initialize")?;
    image::write("Core1.jpg", &image::from_nnf(&nnf, &im_src)?, &params)?;

    let mut d = distance_over_cost(&nnf, &src_border, &target_border, patch).context("D/C")?;
    let rand_nnf = rand_nnf(&nnf, &src_border, &target_border, &mut d, patch).context("Rand")?;
    image::write("Core2.jpg", &image::from_nnf(&rand_nnf, &im_src)?, &params)?;

    let prop_nnf = propagate_nnf(&rand_nnf, &src_border, &target_border, &mut d, patch).context("Prop")?;
    image::write("Core3.jpg", &image::from_nnf(&prop_nnf, &im_src)?, &params)?;
    Ok(())
}

fn distance_over_cost(
    nnf: &Mat,
    src_border: &Mat,
    target_border: &Mat,
    patch: i32,
) -> Result<Vec<f32>> {
    nnf.data_typed::<Point2i>()?
        .iter()
        .enumerate()
        .map(|(idx, p)| {
            let x = idx as i32 % nnf.cols();
            let y = idx as i32 / nnf.cols();
            sum_squared_differences(
                &Mat::roi(src_border, Rect::new(p.x, p.y, patch, patch))?.try_clone()?,
                &Mat::roi(target_border, Rect::new(x, y, patch, patch))?.try_clone()?,
            )
        })
        .collect()
}

fn initialize_nnf(src: &Mat, target: &Mat) -> Result<Mat> {
    let mut rng = rand::rng();
    let mut dst = Mat::new_rows_cols_with_default(
        target.rows(),
        target.cols(),
        opencv::core::CV_32SC2,
        Scalar::default(),
    )?;
    dst.data_typed_mut::<Point2i>()?.iter_mut().for_each(|out| {
        *out = Point2i::new(
            rng.random_range(0..src.cols()),
            rng.random_range(0..src.rows()),
        )
    });
    Ok(dst)
}

fn rand_nnf(
    nnf: &Mat,
    src_border: &Mat,
    target_border: &Mat,
    d: &mut Vec<f32>,
    patch: i32,
) -> Result<Mat> {
    let max_dimension = (nnf.rows() as f32).max(nnf.cols() as f32);
    let mut rng = rand::rng();
    let mut dst = Mat::new_rows_cols_with_default(
        nnf.rows(),
        nnf.cols(),
        opencv::core::CV_32SC2,
        Scalar::default(),
    )?;
    dst.data_typed_mut::<Point2i>()?
        .iter_mut()
        .zip(nnf.data_typed::<Point2i>()?.iter())
        .enumerate()
        .try_for_each(|(idx, (out, p))| -> Result<()> {
            let px = idx as i32 % nnf.cols();
            let py = idx as i32 / nnf.cols();

            let mut best_offset = *p;
            for i in 0..5 {
                let search_radius = max_dimension * (1f32 / 2f32).powi(i);
                let ux = rand_propose(p.x, src_border.cols() - patch, search_radius, &mut rng);
                let uy = rand_propose(p.y, src_border.rows() - patch, search_radius, &mut rng);
                let improved = improved_nnf(
                    &[(
                        Mat::roi(src_border, Rect::new(ux, uy, patch, patch))?, 
                        Mat::roi(target_border, Rect::new(px, py, patch, patch))?, 
                        ux,
                        uy,
                    )],
                    d[idx],
                );
                if let Some((ssd, x, y)) = improved {
                    best_offset = Point2i::new(x, y);
                    d[idx] = ssd;
                }
            }
            *out = best_offset;
            Ok(())
        })?;
    Ok(dst)
}

fn rand_propose(position: i32, max: i32, radius: f32, rng: &mut impl rand::prelude::RngExt) -> i32 {
    (position as f32 + rng.random_range(-1f32..=1f32) * radius).clamp(0.0, max as f32) as i32
}

fn propagate_nnf(
    nnf: &Mat,
    src_border: &Mat,
    target_border: &Mat,
    d: &mut Vec<f32>,
    patch: i32,
) -> Result<Mat> {
    let mut dst = Mat::new_rows_cols_with_default(
        nnf.rows(),
        nnf.cols(),
        opencv::core::CV_32SC2,
        Scalar::default(),
    )?;

    for py in 1..nnf.rows() - 1 {
        for px in 1..nnf.cols() - 1 {
            let idx = py * nnf.cols() + px;

            let left_x = px - 1;
            let up_y = py - 1;

            let mut best_offset = *nnf.at_2d::<Point2i>(py, px)?;
            let improved = improved_nnf(
                &[
                    (
                        Mat::roi(src_border, Rect::new(left_x, py, patch, patch))?,
                        Mat::roi(target_border, Rect::new(px, py, patch, patch))?,
                        left_x,
                        py,
                    ),
                    (
                        Mat::roi(src_border, Rect::new(px, up_y, patch, patch))?,
                        Mat::roi(target_border, Rect::new(px, py, patch, patch))?,
                        px,
                        up_y,
                    ),
                ],
                d[idx as usize],
            );
            if let Some((ssd, x, y)) = improved {
                best_offset = *nnf.at_2d::<Point2i>(y, x)?;
                d[idx as usize] = ssd;
            }
            *dst.at_2d_mut::<Point2i>(py, px)? = best_offset;
        }
    }

    // dst.data_typed_mut::<Point2i>()?
    //     .iter_mut()
    //     .enumerate()
    //     .try_for_each(|(idx, out)| -> Result<()> {
    //         let px = 1 + (idx as i32 % nnf.cols());
    //         let py = 1 + (idx as i32 / nnf.cols());
    //         let left_x = px;
    //         let up_y = py;
    //         let mut best_offset = *nnf.at_2d::<Point2i>(py, px)?;
    //         let improved = improved_nnf(
    //             &[
    //                 (
    //                     Mat::roi(src_border, Rect::new(left_x, py, patch, patch))?,
    //                     Mat::roi(target_border, Rect::new(px, py, patch, patch))?,
    //                     left_x,
    //                     py,
    //                 ),
    //                 (
    //                     Mat::roi(src_border, Rect::new(px, up_y, patch, patch))?,
    //                     Mat::roi(target_border, Rect::new(px, py, patch, patch))?,
    //                     px,
    //                     up_y,
    //                 ),
    //             ],
    //             d[idx],
    //         );
    //         if let Some((ssd, x, y)) = improved {
    //             best_offset = *nnf.at_2d::<Point2i>(y, x)?;
    //             d[idx] = ssd;
    //         }
    //         *out = best_offset;
    //         Ok(())
    //     })?;

    Ok(dst)
}

fn improved_nnf(
    roi_pair: &[(BoxedRef<'_, Mat>, BoxedRef<'_, Mat>, i32, i32)],
    current_ssd: f32,
) -> Option<(f32, i32, i32)> {
    roi_pair.iter().fold(None, |_, (proposed, patch, x, y)| {
        let new_ssd =
            sum_squared_differences(&proposed.try_clone().ok()?, &patch.try_clone().ok()?).ok()?;
        (new_ssd < current_ssd).then_some((new_ssd, *x, *y))
    })
}

fn sum_squared_differences(src_roi: &Mat, target_roi: &Mat) -> Result<f32> {
    Ok(src_roi
        .data_typed::<Vec3b>()?
        .iter()
        .zip(target_roi.data_typed::<Vec3b>()?.iter())
        .fold(0f32, |acc, (a, b)| {
            acc + (a[0] as f32 - b[0] as f32).powi(2)
                + (a[1] as f32 - b[1] as f32).powi(2)
                + (a[2] as f32 - b[2] as f32).powi(2)
        }))
}

// fn propagate_nnf(nnf: &Mat, d: &[f32]) -> Result<Mat> {
//     let rows = nnf.rows();
//     let cols = nnf.cols();
//     let mut dst =
//         Mat::new_rows_cols_with_default(rows, cols, opencv::core::CV_32SC2, Scalar::default())?;
//     dst.data_typed_mut::<Point2i>()?
//         .iter_mut()
//         .enumerate()
//         .try_for_each(|(idx, out)| -> Result<()> {
//             let x = idx as i32 % cols;
//             let y = idx as i32 / cols;
//             let improved = improved_propagate_nnf(idx, x, y, cols, d);
//             *out = *nnf.at_2d::<Point2i>(improved.y, improved.x)?;
//             Ok(())
//         })?;
//     Ok(dst)
// }

// fn improved_propagate_nnf(idx: usize, x: i32, y: i32, cols: i32, d: &[f32]) -> Point2i {
//     let left = coordinates_to_idx(x - 1, y, cols);
//     let up = coordinates_to_idx(x, y - 1, cols);
//     let min = (d[idx]).min(d[left]).min(d[up]);
// let (x, y) = match (d[idx]).min(d[left]).min(d[up]) {
//     n if n == d[left] => (x - 1, y),
//     n if n == d[up] => (x, y - 1),
//     _ => (x, y),
// };
// let (x, y) = if min == d[left] {
//     (x - 1, y)
// } else if min == d[up] {
//     (x, y - 1)
// } else {
//     (x, y)
// };
// Point2i::new(x, y)
// }
// fn coordinates_to_idx(x: i32, y: i32, cols: i32) -> usize {
//     (y.abs() * cols + x.abs()) as usize
// }

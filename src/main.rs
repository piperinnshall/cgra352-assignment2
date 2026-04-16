mod image;

use anyhow::{Context, Result};
use opencv::{
    boxed_ref::BoxedRef,
    core::{
        Mat, MatTrait, MatTraitConst, MatTraitConstManual, MatTraitManual, Point2i, Rect, Scalar,
        Vec3b, Vector,
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
    let iterations = 5;

    let im_src = image::read("Source.jpg")?;
    let im_target = image::read("Target.jpg")?;
    let src_border = image::border(&im_src, patch)?;
    let target_border = image::border(&im_target, patch)?;

    let nnf = initialize_nnf(&im_src, &im_target).context("Initialize")?;
    image::write("Core1.jpg", &image::from_nnf(&nnf, &im_src)?, &params)?;

    let mut d = distance_over_cost(&nnf, &src_border, &target_border, patch).context("D/C")?;
    let rand_nnf = rand_nnf(&nnf, &src_border, &target_border, &mut d, patch).context("Rand")?;
    image::write("Core2.jpg", &image::from_nnf(&rand_nnf, &im_src)?, &params)?;

    let prop_nnf =
        propagate_nnf(&rand_nnf, &src_border, &target_border, &mut d, patch).context("Prop")?;
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
            let mut best_offset = *nnf.at_2d::<Point2i>(py, px)?;
            let left = *dst.at_2d::<Point2i>(py, px - 1)?;
            let up = *dst.at_2d::<Point2i>(py - 1, px)?;
            let improved = improved_nnf(
                &[
                    (
                        Mat::roi(src_border, Rect::new(left.x, left.y, patch, patch))?,
                        Mat::roi(target_border, Rect::new(px, py, patch, patch))?,
                        left.x,
                        left.y,
                    ),
                    (
                        Mat::roi(src_border, Rect::new(up.x, up.y, patch, patch))?,
                        Mat::roi(target_border, Rect::new(px, py, patch, patch))?,
                        up.x,
                        up.y,
                    ),
                ],
                d[idx as usize],
            );
            if let Some((ssd, x, y)) = improved {
                best_offset = Point2i::new(x, y);
                d[idx as usize] = ssd;
            }
            *dst.at_2d_mut::<Point2i>(py, px)? = best_offset;
        }
    }
    Ok(dst)
}

fn improved_nnf(
    roi_pair: &[(BoxedRef<'_, Mat>, BoxedRef<'_, Mat>, i32, i32)],
    current_ssd: f32,
) -> Option<(f32, i32, i32)> {
    let best = roi_pair.iter().fold(None, |best, (proposed, patch, x, y)| {
        let new_ssd =
            sum_squared_differences(&proposed.try_clone().ok()?, &patch.try_clone().ok()?).ok()?;
        match best {
            Some((best_ssd, _, _)) if best_ssd <= new_ssd => best,
            _ => Some((new_ssd, *x, *y)),
        }
    });
    match best {
        Some((ssd, x, y)) if ssd < current_ssd => Some((ssd, x, y)),
        _ => None,
    }
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

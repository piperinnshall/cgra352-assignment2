mod image;

use anyhow::{Context, Result};
use opencv::{
    boxed_ref::BoxedRef,
    core::{
        Mat, MatTraitConst, MatTraitConstManual, MatTraitManual, Point2i, Rect, Scalar, Vec3b,
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
    let im_src = image::read("Source.jpg")?;
    let im_target = image::read("Target.jpg")?;

    let nnf = initialize_nnf(&im_src, &im_target).context("Initialize NNF")?;
    image::write("Core1.jpg", &image::from_nnf(&nnf, &im_src)?, &params)?;

    let patch = 7;
    let src_border = image::border(&im_src, patch)?;
    let target_border = image::border(&im_target, patch)?;
    let mut d = distance_over_cost(&nnf, &src_border, &target_border, patch)?;
    let nnf_rand =
        randomize_nnf(&nnf, &src_border, &target_border, &mut d, patch).context("Randomize NNF")?;
    image::write("Core2.jpg", &image::from_nnf(&nnf_rand, &im_src)?, &params)?;
    Ok(())
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

fn randomize_nnf(
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
            let mut best_offset = *p;
            let px = idx as i32 % nnf.cols();
            let py = idx as i32 / nnf.cols();
            for i in 0..5 {
                let search_radius = max_dimension * (1f32 / 2f32).powi(i);
                let ux = propose_position(p.x, src_border.cols() - patch, search_radius, &mut rng);
                let uy = propose_position(p.y, src_border.rows() - patch, search_radius, &mut rng);
                let improved = improved_nnf(
                    Mat::roi(src_border, Rect::new(ux, uy, patch, patch))?,
                    Mat::roi(target_border, Rect::new(px, py, patch, patch))?,
                    d[idx],
                );
                if let Some(ssd) = improved {
                    best_offset = Point2i::new(ux, uy);
                    d[idx] = ssd;
                }
            }
            Ok(*out = best_offset)
        })?;
    Ok(dst)
}

fn propose_position(
    position: i32,
    max: i32,
    radius: f32,
    rng: &mut impl rand::prelude::RngExt,
) -> i32 {
    (position as f32 + rng.random_range(-1f32..=1f32) * radius).clamp(0.0, max as f32) as i32
}

fn improved_nnf(
    proposed_roi: BoxedRef<'_, Mat>,
    patch_roi: BoxedRef<'_, Mat>,
    current_ssd: f32,
) -> Option<f32> {
    let new_ssd = sum_squared_differences(proposed_roi, patch_roi).ok()?;
    (new_ssd < current_ssd).then_some(new_ssd)
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
            sum_squared_differences(
                Mat::roi(src_border, Rect::new(p.x, p.y, patch, patch))?,
                Mat::roi(
                    target_border,
                    Rect::new(
                        idx as i32 % nnf.cols(),
                        idx as i32 / nnf.cols(),
                        patch,
                        patch,
                    ),
                )?,
            )
        })
        .collect()
}

fn sum_squared_differences(
    src_roi: BoxedRef<'_, Mat>,
    target_roi: BoxedRef<'_, Mat>,
) -> Result<f32> {
    Ok(src_roi
        .try_clone()?
        .data_typed::<Vec3b>()?
        .iter()
        .zip(target_roi.try_clone()?.data_typed::<Vec3b>()?.iter())
        .fold(0f32, |acc, (a, b)| {
            acc + (a[0] as f32 - b[0] as f32).powi(2)
                + (a[1] as f32 - b[1] as f32).powi(2)
                + (a[2] as f32 - b[2] as f32).powi(2)
        }))
}

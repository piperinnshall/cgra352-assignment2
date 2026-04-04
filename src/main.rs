use anyhow::{Context, Error, Result};
use opencv::{
    boxed_ref::BoxedRef,
    core::{
        Mat, MatTraitConst, MatTraitConstManual, MatTraitManual, Point2i, Rect, Scalar, Vec3b,
        Vector,
    },
};
use rand::prelude::RngExt;

fn image_read(path: &str) -> Result<Mat> {
    Ok(opencv::imgcodecs::imread(
        &format!("assets/{}", path),
        opencv::imgcodecs::IMREAD_UNCHANGED,
    )?)
}

fn image_write(path: &str, src: &Mat, params: &Vector<i32>) -> Result<()> {
    Ok(anyhow::ensure!(
        opencv::imgcodecs::imwrite(&format!("assets/{}", path), src, params)?,
        "image write failed: {}",
        path
    ))
}

fn main() -> Result<()> {
    let params = Vector::default();
    core(&params)?;
    Ok(())
}

fn core(params: &Vector<i32>) -> Result<()> {
    let im_src = image_read("Source.jpg")?;
    let im_target = image_read("Target.jpg")?;

    // Initialization
    let nnf = initialize_nnf(&im_src, &im_target).context("Initialize NNF")?;
    image_write("Core1.jpg", &nnf_to_image(&nnf, &im_src)?, &params)?;

    // Random Search
    let patch = 7;
    let d = distance_over_cost(
        &nnf,
        &border(&im_src, patch)?,
        &border(&im_target, patch)?,
        patch,
    )?;
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

fn border(src: &Mat, patch: i32) -> Result<Mat> {
    let pad = (patch as f32 / 2.0).floor() as i32;
    let mut dst = Mat::default();
    opencv::core::copy_make_border(
        src,
        &mut dst,
        pad,
        pad,
        pad,
        pad,
        opencv::core::BORDER_REFLECT_101,
        Scalar::default(),
    )?;
    Ok(dst)
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
        .map(|(i, p)| {
            sum_squared_differences(
                Mat::roi(src_border, Rect::new(p.x, p.y, patch, patch))?,
                Mat::roi(
                    target_border,
                    Rect::new(
                        i as i32 % nnf.cols(),
                        i as i32 / nnf.cols(),
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

fn improve_nnf(nnf: &Mat, _src: &Mat) -> Result<Mat> {
    Ok(Mat::clone(nnf))
}

fn nnf_to_image(nnf: &Mat, src: &Mat) -> Result<Mat> {
    let mut dst = Mat::new_rows_cols_with_default(
        nnf.rows(),
        nnf.cols(),
        opencv::core::CV_8UC3,
        Scalar::default(),
    )?;
    nnf.data_typed::<Point2i>()?
        .iter()
        .zip(dst.data_typed_mut::<Vec3b>()?.iter_mut())
        .try_for_each(|(p, out)| {
            anyhow::ensure!(
                p.x >= 0 && p.y >= 0 && p.x < src.cols() && p.y < src.rows(),
                "Coordinate {:?} is outside of source.",
                p
            );
            let r = (p.x * 255 / src.cols()) as u8;
            let g = (p.y * 255 / src.rows()) as u8;
            let b = 255 - r.max(g);
            Ok(*out = Vec3b::from([b, g, r]))
        })?;
    Ok(dst)
}

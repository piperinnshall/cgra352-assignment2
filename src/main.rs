use anyhow::{Context, Result};
use opencv::core::{
    Mat, MatTraitConst, MatTraitConstManual, MatTraitManual, Point2i, Scalar, Vec3b, Vector,
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
    let core1 = nnf_to_image(&nnf, &im_src).context("NNF to image")?;
    Ok(image_write("Core1.jpg", &core1, &params)?)

    // let nnf = improve_nnf(&nnf, &im_src).context("Improve NNF")?;
}

fn initialize_nnf(src: &Mat, target: &Mat) -> Result<Mat> {
    let mut rng = rand::rng();
    let mut nnf = Mat::new_rows_cols_with_default(
        target.rows(),
        target.cols(),
        opencv::core::CV_32SC2,
        Scalar::default(),
    )?;
    nnf.data_typed_mut::<Point2i>()?.iter_mut().for_each(|out| {
        *out = Point2i::new(
            rng.random_range(0..src.cols()),
            rng.random_range(0..src.rows()),
        )
    });
    Ok(nnf)
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
            let b = 255 - u8::max(r, g);
            Ok(*out = Vec3b::from([b, g, r]))
        })?;
    Ok(dst)
}

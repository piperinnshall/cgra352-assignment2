use anyhow::Result;
use opencv::{
    core::{
        Mat, MatTraitConst, MatTraitConstManual, MatTraitManual, Point2i, Scalar, Vec3b,
        Vector,
    },
};

pub fn read(path: &str) -> Result<Mat> {
    Ok(opencv::imgcodecs::imread(
        &format!("assets/{}", path),
        opencv::imgcodecs::IMREAD_UNCHANGED,
    )?)
}

pub fn write(path: &str, src: &Mat, params: &Vector<i32>) -> Result<()> {
    Ok(anyhow::ensure!(
        opencv::imgcodecs::imwrite(&format!("assets/{}", path), src, params)?,
        "image write failed: {}",
        path
    ))
}

pub fn from_nnf(nnf: &Mat, src: &Mat) -> Result<Mat> {
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

pub fn border(src: &Mat, patch: i32) -> Result<Mat> {
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



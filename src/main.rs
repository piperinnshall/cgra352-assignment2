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

    let nnf = initialize_nnf(&im_src, &im_target).context("Initialize")?;
    image::write("Core1.jpg", &image::from_nnf(&nnf, &im_src)?, &params)?;

    let patch = 7;
    let src_border = image::border(&im_src, patch)?;
    let target_border = image::border(&im_target, patch)?;
    let mut d = distance_over_cost(&nnf, &src_border, &target_border, patch)?;
    let rand_nnf = rand_nnf(&nnf, &src_border, &target_border, &mut d, patch).context("Rand")?;
    image::write("Core2.jpg", &image::from_nnf(&rand_nnf, &im_src)?, &params)?;

    let _ = propagate_nnf(&nnf, &d);
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
            let mut best_offset = *p;
            let px = idx as i32 % nnf.cols();
            let py = idx as i32 / nnf.cols();
            for i in 0..5 {
                let search_radius = max_dimension * (1f32 / 2f32).powi(i);
                let ux = rand_propose(p.x, src_border.cols() - patch, search_radius, &mut rng);
                let uy = rand_propose(p.y, src_border.rows() - patch, search_radius, &mut rng);
                let improved = improved_rand_nnf(
                    Mat::roi(src_border, Rect::new(ux, uy, patch, patch))?,
                    Mat::roi(target_border, Rect::new(px, py, patch, patch))?,
                    d[idx],
                );
                if let Some(ssd) = improved {
                    best_offset = Point2i::new(ux, uy);
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

fn improved_rand_nnf(
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
            let x = idx as i32 % nnf.cols();
            let y = idx as i32 / nnf.cols();
            sum_squared_differences(
                Mat::roi(src_border, Rect::new(p.x, p.y, patch, patch))?,
                Mat::roi(target_border, Rect::new(x, y, patch, patch))?,
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

// Propagation. We attempt to improve f (x, y) using the known
// offsets of f (x − 1, y) and f (x, y − 1), assuming that the patch offsets
// are likely to be the same. For example, if there is a good mapping
// at (x − 1, y), we try to use the translation of that mapping one
// pixel to the right for our mapping at (x, y). Let D(v) denote the
// patch distance (error) between the patch at (x, y) in A and patch
// (x, y) + v in B. We take the new value for f (x, y) to be the arg min
// of {D( f (x, y)), D( f (x − 1, y)), D( f (x, y − 1))}.
// The effect is that if (x, y) has a correct mapping and is in a coherent
// region R, then all of R below and to the right of (x, y) will be
// filled with the correct mapping. Moreover, on even iterations we
// propagate information up and left by examining offsets in reverse
// scan order, using f (x + 1, y) and f (x, y + 1) as our candidate offsets.

// However, propagation with relative coordinates is easier because it just involves trying the
// same offset vector as your adjacent patches, whereas for absolute coordinates it requires an
// additional shift by +1 pixel (it is best to draw on paper if you are confused).

fn propagate_nnf(nnf: &Mat, d: &[f32]) -> Result<Mat> {
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
        .for_each(|(idx, (out, p))| {
            let x = idx as i32 % nnf.cols();
            let y = idx as i32 / nnf.cols();
            // if idx == 462592-1 {
            //     println!("{}, {}, {}, {}", x, y, nnf.cols(), y * nnf.cols() + x)
            // }
            *out = improved_propagate_nnf(idx, x, y, nnf.cols(), d);
        });
    Ok(dst)
}

fn improved_propagate_nnf(idx: usize, x: i32, y: i32, cols: i32, d: &[f32]) -> Point2i {
    let left = (y - 1) * cols + x;
    // let up = coordinates_to_idx(x, y - 1, cols);

    if left as usize == 18446744073709550784 {
        println!("{}, {}, {}, {}", x, y, cols, left as usize);
    }

    let (x, y) = match (d[idx]).min(d[idx]) { //.min(d[left]).min(d[up]) {
        // n if n == d[left] => (x - 1, y),
        // n if n == d[up] => (x, y - 1),
        _ => (x, y),
    };
    Point2i::new(x, y)
}

fn coordinates_to_idx(x: i32, y: i32, cols: i32) -> usize {
    (y * cols + x) as usize
}

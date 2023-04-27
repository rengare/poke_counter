const SAMPLE_TEMPLATE: &str = "./resources/template.png";
const SAMPLE_SOURCE: &str = "./resources/source.png";

const THRESHOLD: f32 = 0.8;

use opencv::{core, highgui, imgcodecs, imgproc, prelude::*};

fn main() -> opencv::Result<()> {
    let template = imgcodecs::imread(SAMPLE_TEMPLATE, imgcodecs::IMREAD_COLOR).unwrap();
    let mut image = imgcodecs::imread(SAMPLE_SOURCE, imgcodecs::IMREAD_COLOR).unwrap();

    // Resize template
    let mut reduced = Mat::default();
    imgproc::resize(
        &template,
        &mut reduced,
        core::Size {
            width: 0,
            height: 0,
        },
        0.5,
        0.5,
        imgproc::INTER_AREA,
    )
    .unwrap();

    let zero = core::Mat::zeros(
        image.rows() - reduced.rows() + 1,
        image.cols() - reduced.cols() + 1,
        core::CV_32FC1,
    )
    .unwrap();

    let mut result = zero.to_mat().unwrap();

    imgproc::match_template(
        &image,      // input image to be searched, 8U or 32F, size W x H
        &reduced,    // template to use, same type as 'image', size w x h
        &mut result, // result image, type 32F, size (W - w + 1) x (H - h + 1)
        imgproc::TM_CCOEFF_NORMED,
        &Mat::default(), // optional mask
    );

    let mut match_count = 0;

    for row in 0..result.rows() {
        for col in 0..result.cols() {
            let value = result.at_2d::<f32>(row, col).unwrap();
            if *value >= THRESHOLD {
                dbg!(value);
                let top_left = core::Rect::new(col, row, reduced.cols(), reduced.rows());

                imgproc::rectangle(
                    &mut image,
                    top_left,
                    core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                    2,
                    8,
                    0,
                )
                .unwrap();

                match_count += 1;
            }
        }
    }

    println!("Match count: {}", match_count);
    highgui::named_window("Matching Result", highgui::WINDOW_AUTOSIZE)?;
    highgui::imshow("Matching Result", &image)?;
    highgui::wait_key(0)?;

    Ok(())
}

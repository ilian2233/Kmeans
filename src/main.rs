use linfa::prelude::*;
use linfa_clustering::{generate_blobs, KMeans};
use ndarray::prelude::*;
use plotters::prelude::*;
use rand::prelude::*;
use rand_isaac::Isaac64Rng;

fn main() {
    //Creates seed for random generation;
    let seed = 42;
    //Creates generator from seed;
    let mut rng = Isaac64Rng::seed_from_u64(seed);
    //Defines Centroids;
    let expected_centroids = array![[1., 3.], [3., 6.], [6., 9.]];
    //Generate 100 random points;
    let data = generate_blobs(100, &expected_centroids, &mut rng);
    //Converts random points to DataSet;
    let dataset = DatasetBase::from(data.clone());
    //Defines number of clusters;
    let n_clusters = 3;
    //Loads the models with clusters, Datasets and set number of iterations;
    let model = KMeans::params_with_rng(n_clusters, rng)
        .max_n_iterations(200)
        .tolerance(1e-2)
        .fit(&dataset)
        .expect("KMeans fitted");
    //Uses the trained model to predict on the same dataset;
    let dataset = model.predict(dataset);

    //Fancy drawing stuff for creating png;
    {
        let root = BitMapBackend::new("kmeans.png", (600, 400)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let x_lim = 0.0..10.0f64;
        let y_lim = 0.0..10.0f64;

        let mut ctx = ChartBuilder::on(&root)
            .set_label_area_size(LabelAreaPosition::Left, 40) // Put in some margins
            .set_label_area_size(LabelAreaPosition::Right, 40)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            .caption("KMeans Demo", ("sans-serif", 25)) // Set a caption and font
            .build_cartesian_2d(x_lim, y_lim)
            .expect("Couldn't build our ChartBuilder");

        ctx.configure_mesh().draw().unwrap();
        let root_area = ctx.plotting_area();

        for i in 0..dataset.records.shape()[0] {
            let coordinates = dataset.records.slice(s![i, 0..2]);

            let point = match dataset.targets[i] {
                0 => Circle::new(
                    (coordinates[0], coordinates[1]),
                    3,
                    ShapeStyle::from(&RED).filled(),
                ),
                1 => Circle::new(
                    (coordinates[0], coordinates[1]),
                    3,
                    ShapeStyle::from(&GREEN).filled(),
                ),

                2 => Circle::new(
                    (coordinates[0], coordinates[1]),
                    3,
                    ShapeStyle::from(&BLUE).filled(),
                ),
                // Making sure our pattern-matching is exhaustive
                _ => Circle::new(
                    (coordinates[0], coordinates[1]),
                    3,
                    ShapeStyle::from(&BLACK).filled(),
                ),
            };

            root_area
                .draw(&point)
                .expect("An error occurred while drawing the point!");
        }
    }
}

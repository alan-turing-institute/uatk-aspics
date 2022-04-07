//! This is the command-line interface to ASPICS.

use std::collections::BTreeMap;

use anyhow::Result;
use clap::Parser;
use fs_err::File;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Deserialize;
use tracing::{info, info_span};

use aspics::utilities;
use aspics::{protobuf, Input, Model, Population, MSOA};

// When running on all MSOAs, start with this many cases
const DEFAULT_CASES_PER_MSOA: usize = 5;

#[tokio::main]
async fn main() -> Result<()> {
    aspics::tracing_span_tree::SpanTree::new().enable();

    let args = Args::parse();

    let mut rng = if let Some(seed) = args.rng_seed {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_entropy()
    };

    match args.action {
        Action::Init {
            region,
            no_commuting,
        } => {
            let _s = info_span!("initialisation", ?region).entered();
            let input = region.to_input(!no_commuting).await?;
            let population = Population::create(input, &mut rng).await?;

            // First clear the target directory
            let target_dir = format!("data/processed_data/{:?}", region);
            // Ignore errors if this directory doesn't even exist
            let _ = fs_err::remove_dir_all(&target_dir);

            info!("By the end, {}", utilities::memory_usage());

            {
                let output = format!("{target_dir}/synthpop.pb");
                let _s = info_span!("Writing protobuf to", ?output).entered();
                protobuf::convert_to_pb(&population, output)?;
            }
        }
        Action::RunModel { region } => {
            info!("Loading population");
            let population = utilities::read_binary::<Population>(format!(
                "data/processed_data/{:?}/rust_cache.bin",
                region
            ))?;
            let mut model = Model::new(population, rng)?;
            model.run()?;
        }
    }

    Ok(())
}

#[derive(Parser)]
#[clap(about, version, author)]
struct Args {
    #[clap(subcommand)]
    action: Action,
    /// By default, the output will be different every time the tool is run, based on a different
    /// random number generator seed. Specify this to get deterministic behavior, given the same
    /// input.
    #[clap(long)]
    rng_seed: Option<u64>,
}

#[derive(clap::ArgEnum, Clone, Copy, Debug)]
/// Which counties to operate on
enum Region {
    WestYorkshireSmall,
    WestYorkshireLarge,
    Devon,
    TwoCounties,
    National,
}

#[derive(clap::Subcommand, Clone)]
enum Action {
    /// Import raw data and build an activity model for a region
    Init {
        #[clap(arg_enum)]
        region: Region,
        #[clap(long)]
        no_commuting: bool,
    },
    /// Run the model, for a fixed number of days
    RunModel {
        #[clap(arg_enum)]
        region: Region,
    },
}

impl Region {
    async fn to_input(self, enable_commuting: bool) -> Result<Input> {
        let mut input = Input {
            enable_commuting,
            initial_cases_per_msoa: BTreeMap::new(),
        };

        // Determine the MSOAs to operate on using CSV files from the original repo
        let csv_input = match self {
            Region::National => {
                for msoa in MSOA::all_msoas_nationally().await? {
                    input
                        .initial_cases_per_msoa
                        .insert(msoa, DEFAULT_CASES_PER_MSOA);
                }
                return Ok(input);
            }
            _ => format!("Input_{:?}.csv", self),
        };
        let csv_path = format!("../config/{}", csv_input);
        for rec in csv::Reader::from_reader(File::open(csv_path)?).deserialize() {
            let rec: InitialCaseRow = rec?;
            input.initial_cases_per_msoa.insert(rec.msoa, rec.cases);
        }
        Ok(input)
    }
}

#[derive(Deserialize)]
struct InitialCaseRow {
    #[serde(rename = "MSOA11CD")]
    msoa: MSOA,
    cases: usize,
}

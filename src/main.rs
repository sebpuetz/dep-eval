use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::collections::HashMap;
use std::fmt::Display;
use std::fmt;
use std::hash::Hash;
use std::io::Write;

use clap::{App, AppSettings, Arg, ArgMatches};
use conllx::io::{Reader, ReadSentence};
use conllx::token::Token;
use failure::{Error};
use itertools::Itertools;
use stdinout::OrExit;

pub fn main() -> Result<(), Error> {
    let matches = parse_args();
    let val_path = matches
        .value_of(VALIDATION)
        .or_exit("Missing input path", 1);
    let val_file = File::open(val_path).or_exit("Can't open validation file.", 1);
    let mut val_reader = Reader::new(BufReader::new(val_file));

    let pred_path = matches
        .value_of(PREDICTION)
        .or_exit("Missing input path", 1);
    let pred_file = File::open(pred_path)?;
    let mut pred_reader = Reader::new(BufReader::new(pred_file));

    let mut deprel_confusion = Confusion::<String>::new("Deprels");
    let mut distance_confusion = Confusion::<usize>::new("Dists");

    let mut correct_head = 0;
    let mut correct_head_label = 0;
    let mut total = 0;

    while let (Ok(Some(val_sentence)), Ok(Some(pred_sentence))) = (val_reader.read_sentence(), pred_reader.read_sentence()) {
        assert_eq!(val_sentence.len(), pred_sentence.len());
        for (idx, (val_token, pred_token)) in val_sentence
            .iter()
            .filter_map(|t| t.token())
            .zip(pred_sentence.iter().filter_map(|t| t.token()))
            .enumerate() {
            assert_eq!(val_token.form(), pred_token.form());
            let idx = idx+1 ;
            let val_triple = val_sentence.dep_graph().head(idx).unwrap();
            let val_head = val_triple.head();
            let val_dist = i64::abs(val_head as i64 - idx as i64) as usize;
            let val_rel = val_triple.relation().unwrap();
            let pred_triple = pred_sentence.dep_graph().head(idx).unwrap();;
            let pred_head = pred_triple.head();
            let pred_dist = i64::abs(pred_head as i64 - idx as i64) as usize;
            let pred_rel = pred_triple.relation().unwrap();
            distance_confusion.insert(val_dist, pred_dist);

            deprel_confusion.insert(val_rel, pred_rel);

            correct_head += (pred_head == val_head) as usize;
            correct_head_label += (pred_triple == val_triple) as usize;
            total += 1;
        }
    }
    println!("UAS: {:.4}", correct_head as f32 / total as f32);
    println!("LAS: {:.4}", correct_head_label as f32 / total as f32);

    if let Some(file_name) = matches.value_of(DEPREL_CONFUSION) {
        let out = File::create(file_name).unwrap();
        let mut writer = BufWriter::new(out);
        write!(writer, "{}", deprel_confusion).unwrap();
    }
    if let Some(file_name) = matches.value_of(DEPREL_ACCURACIES) {
        let out = File::create(file_name).unwrap();
        let mut writer = BufWriter::new(out);
        deprel_confusion.write_accuracies(&mut writer).unwrap();
    }

    if let Some(file_name) = matches.value_of(DISTANCE_CONFUSION) {
        let out = File::create(file_name).unwrap();
        let mut writer = BufWriter::new(out);
        write!(writer, "{}", distance_confusion).unwrap();
//        write!(writer, "{}", deprel_confusion).unwrap();
    }
    if let Some(file_name) = matches.value_of(DISTANCE_ACCURACIES) {
        let out = File::create(file_name).unwrap();
        let mut writer = BufWriter::new(out);
        distance_confusion.write_accuracies(&mut writer).unwrap();
    }
    Ok(())
}

static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
];

// Argument constants
static VALIDATION: &str = "VALIDATION";
static PREDICTION: &str = "PREDICTION";
static CLAUSE_IDS: &str = "clause_ids";
static NO_FIELDS: &str = "no_fields";
static DEPREL_CONFUSION: &str = "deprel_confusion";
static DEPREL_ACCURACIES: &str = "deprel_accuracies";
static DISTANCE_ACCURACIES: &str = "distance_confusion";
static DISTANCE_CONFUSION: &str = "distance_accuracies";
static NO_RELS: &str = "no_rels";
static FIELD_FEATURE_NAME: &str  = "tf_feature";

fn parse_args() -> ArgMatches<'static> {
    App::new("reduce-ptb")
        .settings(DEFAULT_CLAP_SETTINGS)
        .arg(
            Arg::with_name(VALIDATION)
                .help("VALIDATION file")
                .index(1)
                .required(true),
        )
        .arg(
            Arg::with_name(PREDICTION)
                .index(2)
                .help("PREDICTION")
                .required(true),
        )
        .arg(
            Arg::with_name(DEPREL_CONFUSION)
                .takes_value(true)
                .long(DEPREL_CONFUSION)
                .help("print deprel confusion matrix to file")
        )
        .arg(
            Arg::with_name(DISTANCE_CONFUSION)
                .takes_value(true)
                .long(DISTANCE_CONFUSION)
                .help("print DISTANCE_CONFUSION matrix to file")
        )
        .arg(
            Arg::with_name(DISTANCE_ACCURACIES)
                .takes_value(true)
                .long(DISTANCE_ACCURACIES)
                .help("print DISTANCE_ACCURACIES to file")
        )
        .arg(
            Arg::with_name(DEPREL_ACCURACIES)
                .takes_value(true)
                .long(DEPREL_ACCURACIES)
                .help("print DISTANCE_ACCURACIES to file")
        )
        .arg(
            Arg::with_name(CLAUSE_IDS)
                .long(CLAUSE_IDS)
                .help("Use clause IDs to derive rel predictions.")
        )
        .arg(
            Arg::with_name(NO_FIELDS)
                .long(NO_FIELDS)
                .help("Don't evaluate topological fields")
                .conflicts_with(NO_RELS)
        )
        .arg(
            Arg::with_name(NO_RELS)
                .long(NO_RELS)
                .help("Don't evaluate relations")
                .conflicts_with(NO_FIELDS)
        )
        .arg(
            Arg::with_name(FIELD_FEATURE_NAME)
                .long(FIELD_FEATURE_NAME)
                .help("Use other field feature name for pred file. (Default p_tf)")
                .takes_value(true)
                .conflicts_with(NO_FIELDS)
        )
        .get_matches()
}

pub trait GetFeature {
    fn get_feature(&self, name: &str) -> Option<&str>;
}

impl GetFeature for Token {
    fn get_feature(&self, name: &str) -> Option<&str> {
        if let Some(features) = self.features() {
            if let Some(feature) = features.as_map().get(name) {
                return feature.as_ref().map(|f| f.as_str())
            }
        }
        None
    }
}

pub struct Confusion<V> {
    confusion: Vec<Vec<usize>>,
    numberer: Numberer<V>,
    name: String,
}

impl<V> Confusion<V> where V: Clone + Hash + Eq {
    pub fn new(name: impl Into<String>) -> Self {
        Confusion {
            confusion: Vec::new(),
            numberer: Numberer::new(),
            name: name.into(),
        }
    }

    pub fn insert<S>(&mut self, target: S, prediction: S) where S: Into<V> {
        let target_idx = self.numberer.number(target);
        let pred_idx = self.numberer.number(prediction);
        while target_idx >= self.confusion.len() || pred_idx >= self.confusion.len() {
            self.confusion.push(vec![0; self.confusion.len()]);
            self.confusion
                .iter_mut()
                .for_each(|row| row.push(0));
        }
        self.confusion[target_idx][pred_idx] += 1;
    }
}
impl<V> Confusion<V> {
    pub fn numberer(&self) -> &Numberer<V> {
        &self.numberer
    }
}

impl<V> Confusion<V> where V: ToString {

    fn write_accuracies(&self, mut w: impl Write) -> Result<(), Error> {
        for (idx, item) in self.numberer.idx2val.iter().map(V::to_string).enumerate() {
            let row = &self.confusion[idx];
            let correct = row[idx];
            let total = row.iter().sum::<usize>();
            let acc = correct as f32 / total as f32;
            writeln!(w, "{}\t{}\t{:.04}", item, total, acc)?;
        }
        Ok(())
    }

    pub fn write_to_file(&self, mut w: impl Write, sep: &str) -> Result<(), Error> {
        writeln!(w, "{}", self.numberer.idx2val.iter().map(ToString::to_string).join(sep))?;
        for i in 0..self.confusion.len() {
            writeln!(w, "{}", self.confusion[i].iter().map(|n| n.to_string()).join(sep))?;
        }
        Ok(())
    }

}

impl<V> Display for Confusion<V> where V: ToString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{}\t{}", self.name, self.numberer.idx2val.iter().map(ToString::to_string).join("\t"))?;
        let mut total_correct = 0;
        let mut full_total = 0;
        for (idx, val) in self.numberer.idx2val.iter().enumerate() {
            let row = &self.confusion[idx];
            let correct = row[idx];
            total_correct += correct;
            let total = row.iter().sum::<usize>();
            full_total += total;
            let acc = correct as f32 / total as f32;
            writeln!(f, "{}\t{}\t{:.4}", val.to_string(), self.confusion[idx].iter().map(|n| n.to_string()).join("\t"), acc)?;
        }
        let mut delim = String::new();
        let mut precs = String::new();
        for i in 0..self.confusion.len() {
            let mut false_pos = 0;
            for j in 0..self.confusion.len() {
                if j == i {
                    continue
                }
                false_pos += self.confusion[j][i]
            }
            let prec = self.confusion[i][i] as f32 / (self.confusion[i][i] + false_pos) as f32;
            precs.push_str(&format!("\t{:.4}", prec));
            delim.push_str("\t____");
        }
        writeln!(f, "{}", delim)?;
        writeln!(f, "{}", precs)?;
        let acc = total_correct as f32 / full_total as f32;
        writeln!(f, "acc: {:.4}", acc)?;
        Ok(())
    }
}

pub struct Numberer<V>{
    val2idx: HashMap<V, usize>,
    idx2val: Vec<V>,
}

impl<V> Numberer<V> where V: Clone + Hash + Eq {
    pub fn new() -> Self {
        Numberer {
            val2idx: HashMap::new(),
            idx2val: Vec::new(),
        }
    }

    fn number<S>(&mut self, val: S) -> usize where S: Into<V> {
        let val = val.into();
        if let Some(idx) = self.val2idx.get(&val) {
            *idx
        } else {
            let n_vals = self.val2idx.len();
            self.val2idx.insert(val.clone(), n_vals);
            self.idx2val.push(val);
            n_vals
        }
    }

    pub fn get_number(&self, val: &V) -> Option<usize> {
        self.val2idx.get(val).map(|idx| *idx)
    }
}

impl<V> Numberer<V> {
    pub fn len(&self) -> usize {
        self.idx2val.len()
    }

    pub fn is_empty(&self) -> bool {
        self.idx2val.is_empty()
    }

    pub fn get_val(&self, idx: usize) -> Option<&V> {
        self.idx2val.get(idx)
    }
}
package benchmark

import treebased.api.CausalForest
import treebased.config.ExactCausalForestConfig
import treebased.core.domain.data.CausalObservation

import java.io.{BufferedWriter, FileWriter}
import java.nio.file.{Files, Paths}
import scala.util.Random

/**
 * Benchmark entry point.
 *
 * Generates a synthetic RCT dataset, trains a heavy CausalForest, writes
 * predictions and timing to data/ so compare.py can pick them up.
 *
 * Usage:  sbt "runMain benchmark.ScalaBenchmark"
 *
 * Data-generating process (all features continuous):
 *   features ~ N(offset_i, 1)   for i in 0..N_FEATURES-1
 *   treatment ~ Bernoulli(0.5)
 *   label = f0 + 0.5*f1 + f2*(-2)*treatment + N(0,1)
 *   True HTE: τ(x) = f2 × -2   (heterogeneous, driven only by feature index 2)
 */
object ScalaBenchmark {

  // ── Configuration ────────────────────────────────────────────────────────────

  val N_TRAIN                  = 20_000
  val N_TEST                   = 5_000
  val N_FEATURES               = 10
  val N_TREES                  = 200
  val MAX_DEPTH                = 15
  val MIN_NODE_SIZE_PER_GROUP  = 5
  // sqrt(10) ≈ 3.16 → use 4; standard causal-forest choice
  val MAX_FEATURES_FOR_SPLIT   = math.max(1, math.sqrt(N_FEATURES).toInt + 1)
  val SUBSAMPLE_RATIO          = 0.8
  val SEED                     = 42L
  val DATA_DIR                 = "data"

  // ── Data generation ──────────────────────────────────────────────────────────

  def trueHTE(features: Array[Double]): Double = features(2) * -2.0

  def generateData(n: Int, rng: Random): Array[CausalObservation] =
    Array.fill(n) {
      // Feature offsets mirror the 3-feature pattern from generatePoints.scala
      val features  = Array.tabulate(N_FEATURES)(i => rng.nextGaussian() + (i % 3) * 0.2)
      val treatment = rng.nextInt(2).toDouble
      val noise     = rng.nextGaussian()
      val label     = features(0) + features(1) * 0.5 + features(2) * -2.0 * treatment + noise
      CausalObservation(features, 1.0, label, treatment)
    }

  // ── CSV helpers ──────────────────────────────────────────────────────────────

  def writeCsv(path: String, data: Array[CausalObservation]): Unit = {
    val header = (1 to N_FEATURES).map(i => s"x$i").mkString(",") +
      ",treatment,weight,label,true_hte"
    val bw = new BufferedWriter(new FileWriter(path))
    try {
      bw.write(header); bw.newLine()
      data.foreach { dp =>
        val hte = trueHTE(dp.features)
        bw.write(s"${dp.features.mkString(",")},${dp.w.toInt},${dp.weight},${dp.y},$hte")
        bw.newLine()
      }
    } finally bw.close()
  }

  def writePredictions(path: String, preds: Array[Double], trueHTEs: Array[Double]): Unit = {
    val bw = new BufferedWriter(new FileWriter(path))
    try {
      bw.write("scala_pred,true_hte"); bw.newLine()
      preds.zip(trueHTEs).foreach { case (p, t) =>
        bw.write(s"$p,$t"); bw.newLine()
      }
    } finally bw.close()
  }

  def writeTiming(path: String, trainMs: Double, predMs: Double): Unit = {
    val bw = new BufferedWriter(new FileWriter(path))
    try {
      bw.write("phase,milliseconds"); bw.newLine()
      bw.write(s"training,$trainMs");  bw.newLine()
      bw.write(s"prediction,$predMs"); bw.newLine()
    } finally bw.close()
  }

  // ── Metrics ──────────────────────────────────────────────────────────────────

  def mse(preds: Array[Double], targets: Array[Double]): Double =
    preds.zip(targets).map { case (p, t) => (p - t) * (p - t) }.sum / preds.length

  def r2(preds: Array[Double], targets: Array[Double]): Double = {
    val err  = mse(preds, targets) * preds.length
    val mean = targets.sum / targets.length
    val sst  = targets.map(t => (t - mean) * (t - mean)).sum
    1.0 - err / sst
  }

  // ── Main ─────────────────────────────────────────────────────────────────────

  def main(args: Array[String]): Unit = {

    Files.createDirectories(Paths.get(DATA_DIR))

    // ── Generate data ──────────────────────────────────────────────────────────
    println(s"Generating data: $N_TRAIN train + $N_TEST test, $N_FEATURES features, seed=$SEED")
    val rng     = new Random(SEED)
    val allData = generateData(N_TRAIN + N_TEST, rng)
    val (trainData, testData) = allData.splitAt(N_TRAIN)

    println(s"Writing CSVs to $DATA_DIR/ ...")
    writeCsv(s"$DATA_DIR/train.csv", trainData)
    writeCsv(s"$DATA_DIR/test.csv",  testData)

    // ── Forest config ──────────────────────────────────────────────────────────
    val config = ExactCausalForestConfig(
      maxDepth                     = MAX_DEPTH,
      nTrees                       = N_TREES,
      maxFeaturesForSplit          = MAX_FEATURES_FOR_SPLIT,
      minNodeSizePerTreatmentGroup = MIN_NODE_SIZE_PER_GROUP,
      subsampleRatio               = SUBSAMPLE_RATIO,
      seed                         = SEED
    )

    println(
      s"\nConfig: nTrees=$N_TREES  maxDepth=$MAX_DEPTH  " +
      s"maxFeaturesForSplit=$MAX_FEATURES_FOR_SPLIT  " +
      s"minNodeSizePerGroup=$MIN_NODE_SIZE_PER_GROUP"
    )

    // ── Training ───────────────────────────────────────────────────────────────
    println(s"\nTraining Scala CausalForest...")
    val t0 = System.nanoTime()
    val forest = CausalForest.train(trainData, config)
    val trainMs = (System.nanoTime() - t0) / 1e6
    println(f"  Training time : $trainMs%,.1f ms")

    // ── Prediction ─────────────────────────────────────────────────────────────
    println(s"Predicting on $N_TEST test points...")
    val t1    = System.nanoTime()
    val preds = forest.predict(testData.map(_.features))
    val predMs = (System.nanoTime() - t1) / 1e6
    println(f"  Prediction time: $predMs%,.1f ms")

    // ── Metrics ────────────────────────────────────────────────────────────────
    val trueHTEs = testData.map(dp => trueHTE(dp.features))
    val testMse  = mse(preds, trueHTEs)
    val testR2   = r2(preds, trueHTEs)
    println(f"\n  Test MSE : $testMse%.4f")
    println(f"  Test R²  : $testR2%.4f")

    // ── Write outputs ──────────────────────────────────────────────────────────
    writePredictions(s"$DATA_DIR/scala_predictions.csv", preds, trueHTEs)
    writeTiming(s"$DATA_DIR/scala_timing.csv", trainMs, predMs)

    println(s"\nOutputs written to $DATA_DIR/")
    println("Next: run `python3 compare.py` (or `bash run.sh`)")
  }
}

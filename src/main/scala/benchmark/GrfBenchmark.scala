package benchmark

import treebased.api.{CausalForest, GeneralizedRandomForest}
import treebased.config.{CausalForestConfig, ExactCausalForestConfig, GradientSplitParams, RegressionForestConfig}
import treebased.estimand.Centering
import treebased.core.domain.data.{CausalObservation, Observation}

import java.io.{BufferedWriter, FileWriter}
import java.nio.file.{Files, Paths}
import scala.util.Random

/**
 * Speed and accuracy benchmark for every forest in this library.
 *
 *   1. CausalForest.train              — Athey & Wager (2018) exact-splitting causal forest
 *   2. GeneralizedRandomForest.causal  — gradient-splitting causal forest (centering OFF)
 *   3. GeneralizedRandomForest.causal  — gradient-splitting causal forest (OOB cross-fit centering)
 *   4. GeneralizedRandomForest.regression — gradient-splitting regression forest
 *
 * Writes the shared datasets and a tidy results CSV to data/ so any other
 * implementation can be timed on byte-identical data and append its own rows to the
 * same results file.
 *
 * Usage:  sbt "runMain benchmark.GrfBenchmark"
 */
object GrfBenchmark {

  // ── Shared configuration ────────────────────────────────────────────────────
  val N_TRAINS   = Array(5_000, 20_000)
  val N_TEST     = 5_000
  val N_FEATURES = 10
  val N_TREES    = 500
  val MAX_DEPTH  = 30            // large: let min-node-size be the binding stop rule
  val MIN_LEAF   = 5             // min samples per leaf / per treatment arm
  val MTRY       = N_FEATURES    // features tried per split (all p=10 features here)
  val SAMPLE_FRAC = 0.5          // subsample fraction per tree (standard honest-forest default)
  val HONEST_FRAC = 0.5          // fraction of the subsample held out for leaf estimation
  val SEED       = 42L
  val REPS       = 3
  val DATA_DIR   = "data"

  def trueHTE(f: Array[Double]): Double = f(2) * -2.0

  def generateData(n: Int, rng: Random): Array[CausalObservation] =
    Array.fill(n) {
      val features  = Array.tabulate(N_FEATURES)(i => rng.nextGaussian() + (i % 3) * 0.2)
      val treatment = rng.nextInt(2).toDouble
      val noise     = rng.nextGaussian()
      val label     = features(0) + features(1) * 0.5 + features(2) * -2.0 * treatment + noise
      CausalObservation(features, 1.0, label, treatment)
    }

  def writeCsv(path: String, data: Array[CausalObservation]): Unit = {
    val header = (1 to N_FEATURES).map(i => s"x$i").mkString(",") + ",treatment,weight,label,true_hte"
    val bw = new BufferedWriter(new FileWriter(path))
    try {
      bw.write(header); bw.newLine()
      data.foreach { dp =>
        bw.write(s"${dp.features.mkString(",")},${dp.w.toInt},${dp.weight},${dp.y},${trueHTE(dp.features)}")
        bw.newLine()
      }
    } finally bw.close()
  }

  def rmse(preds: Array[Double], targets: Array[Double]): Double =
    math.sqrt(preds.zip(targets).map { case (p, t) => (p - t) * (p - t) }.sum / preds.length)

  def median(xs: Seq[Double]): Double = {
    val s = xs.sorted
    val n = s.length
    if (n % 2 == 1) s(n / 2) else (s(n / 2 - 1) + s(n / 2)) / 2.0
  }

  private def exactCausalConfig(nTrain: Int): ExactCausalForestConfig = ExactCausalForestConfig(
    maxDepth                     = MAX_DEPTH,
    nTrees                       = N_TREES,
    maxFeaturesForSplit          = MTRY,
    minNodeSize                  = MIN_LEAF,
    minNodeSizePerTreatmentGroup = MIN_LEAF,
    predictionDataRatio          = HONEST_FRAC,
    subsampleRatio               = SAMPLE_FRAC,
    seed                         = SEED
  )

  private def causalForestConfig(nTrain: Int): CausalForestConfig = CausalForestConfig(
    maxDepth                     = MAX_DEPTH,
    nTrees                       = N_TREES,
    maxFeaturesForSplit          = MTRY,
    minNodeSize                  = MIN_LEAF,
    minNodeSizePerTreatmentGroup = MIN_LEAF,
    predictionDataRatio          = HONEST_FRAC,
    subsampleRatio               = SAMPLE_FRAC,
    seed                         = SEED
  )

  private def regressionForestConfig(nTrain: Int): RegressionForestConfig = RegressionForestConfig(
    maxDepth                     = MAX_DEPTH,
    nTrees                       = N_TREES,
    maxFeaturesForSplit          = MTRY,
    minNodeSize                  = MIN_LEAF,
    predictionDataRatio          = HONEST_FRAC,
    subsampleRatio               = SAMPLE_FRAC,
    seed                         = SEED
  )

  /** One timed (train, predict) pair. Returns (trainMs, predictMs, rmse). */
  private def timeRun(
    label: String,
    train: () => (Array[Array[Double]] => Array[Double]),
    testX: Array[Array[Double]],
    truth: Array[Double]
  ): (Double, Double, Double) = {
    val t0 = System.nanoTime()
    val predictFn = train()
    val trainMs = (System.nanoTime() - t0) / 1e6
    val t1 = System.nanoTime()
    val preds = predictFn(testX)
    val predMs = (System.nanoTime() - t1) / 1e6
    (trainMs, predMs, rmse(preds, truth))
  }

  def main(args: Array[String]): Unit = {
    Files.createDirectories(Paths.get(DATA_DIR))
    val rng = new Random(SEED)

    // Shared test set (fixed across every training size).
    val testData = generateData(N_TEST, rng)
    writeCsv(s"$DATA_DIR/test.csv", testData)
    val testX    = testData.map(_.features)
    val trueTau  = testData.map(d => trueHTE(d.features))
    val trueMean = testData.map(d => d.features(0) + d.features(1) * 0.5 + d.features(2) * -1.0) // E[Y|X] at T~Bern(.5)

    val results = new BufferedWriter(new FileWriter(s"$DATA_DIR/bench_results.csv"))
    results.write("framework,model,group,n_train,n_trees,rep,train_ms,predict_ms,rmse")
    results.newLine()

    def record(model: String, group: String, nTrain: Int, rep: Int, r: (Double, Double, Double)): Unit = {
      results.write(f"scala-thisproject,$model,$group,$nTrain,$N_TREES,$rep,${r._1}%.3f,${r._2}%.3f,${r._3}%.5f")
      results.newLine(); results.flush()
      println(f"  [$model%-28s n=$nTrain%6d rep=$rep] train=${r._1}%,10.1f ms  predict=${r._2}%,8.1f ms  rmse=${r._3}%.4f")
    }

    println(s"JVM: ${System.getProperty("java.version")}  cores=${Runtime.getRuntime.availableProcessors}")
    println(s"Config: nTrees=$N_TREES maxDepth=$MAX_DEPTH minLeaf=$MIN_LEAF mtry=$MTRY sampleFrac=$SAMPLE_FRAC honestFrac=$HONEST_FRAC\n")

    for (nTrain <- N_TRAINS) {
      println(s"════════ N_train = $nTrain ════════")
      val trainRng   = new Random(SEED + nTrain)
      val trainCausal = generateData(nTrain, trainRng)
      val trainReg    = trainCausal.map(d => Observation(d.features, d.weight, d.y))
      writeCsv(s"$DATA_DIR/train_$nTrain.csv", trainCausal)

      val exactCfg      = exactCausalConfig(nTrain)
      val causalCfg     = causalForestConfig(nTrain)
      val regressionCfg = regressionForestConfig(nTrain)
      val kernelCfg     = regressionCfg.copy(kernelPrediction = true)

      // Warm-up (JIT) — one untimed pass of each model at this size.
      println("  warming up JIT ...")
      CausalForest.train(trainCausal, exactCfg).predict(testX)
      GeneralizedRandomForest.causal(trainCausal, causalCfg.copy(centering = Centering.Off)).predict(testX)
      GeneralizedRandomForest.regression(trainReg, regressionCfg).predict(testX)

      val lightNuisance = regressionForestConfig(nTrain).copy(nTrees = 100, maxDepth = 12)
      // Ablation: the plain split criterion — no child-balance rule, no stabilization.
      val noStabCfg     = causalCfg.copy(splitTuning = GradientSplitParams(splitBalanceAlpha = 0.0, stabilizeSplits = false))

      for (rep <- 1 to REPS) {
        // ── raw-causal ───────────────────────────────────────────────────────
        record("CausalForest (exact Athey-Wager 2018)", "raw-causal", nTrain, rep,
          timeRun("cf-exact", () => {
            val f = CausalForest.train(trainCausal, exactCfg)
            (x: Array[Array[Double]]) => f.predict(x)
          }, testX, trueTau))

        // Default configuration: full-mtry, midpoint thresholds, alpha=0.05, stabilize on,
        // kernel prediction on.
        record("GRF.causal (no-centering)", "raw-causal", nTrain, rep,
          timeRun("grf-causal-off", () => {
            val f = GeneralizedRandomForest.causal(trainCausal, causalCfg.copy(centering = Centering.Off))
            (x: Array[Array[Double]]) => f.predict(x)
          }, testX, trueTau))

        // Same, but averaging per-tree leaf slopes instead of the alpha-weighted solve.
        record("GRF.causal (no-centering / per-tree avg)", "raw-causal", nTrain, rep,
          timeRun("grf-causal-off-avg", () => {
            val f = GeneralizedRandomForest.causal(trainCausal, causalCfg.copy(centering = Centering.Off, kernelPrediction = false))
            (x: Array[Array[Double]]) => f.predict(x)
          }, testX, trueTau))

        // ── centered-causal ──────────────────────────────────────────────────
        record("GRF.causal (OOB crossfit)", "centered-causal", nTrain, rep,
          timeRun("grf-causal-cf", () => {
            val f = GeneralizedRandomForest.causal(trainCausal, causalCfg.copy(centering = Centering.CrossFit()))
            (x: Array[Array[Double]]) => f.predict(x)
          }, testX, trueTau))

        record("GRF.causal (OOB crossfit / per-tree avg)", "centered-causal", nTrain, rep,
          timeRun("grf-causal-cf-avg", () => {
            val f = GeneralizedRandomForest.causal(trainCausal, causalCfg.copy(centering = Centering.CrossFit(), kernelPrediction = false))
            (x: Array[Array[Double]]) => f.predict(x)
          }, testX, trueTau))

        record("GRF.causal (OOB crossfit / no alpha+stabilize)", "centered-causal", nTrain, rep,
          timeRun("grf-causal-cf-nostab", () => {
            val f = GeneralizedRandomForest.causal(trainCausal, noStabCfg.copy(centering = Centering.CrossFit()))
            (x: Array[Array[Double]]) => f.predict(x)
          }, testX, trueTau))

        record("GRF.causal (OOB crossfit / light nuisance)", "centered-causal", nTrain, rep,
          timeRun("grf-causal-cf-light", () => {
            val f = GeneralizedRandomForest.causal(trainCausal, causalCfg.copy(centering = Centering.CrossFit(Some(lightNuisance))))
            (x: Array[Array[Double]]) => f.predict(x)
          }, testX, trueTau))

        // ── regression ───────────────────────────────────────────────────────
        record("GRF.regression", "regression", nTrain, rep,
          timeRun("grf-reg", () => {
            val f = GeneralizedRandomForest.regression(trainReg, regressionCfg)
            (x: Array[Array[Double]]) => f.predict(x)
          }, testX, trueMean))

        record("GRF.regression (kernel)", "regression", nTrain, rep,
          timeRun("grf-reg-kernel", () => {
            val f = GeneralizedRandomForest.regression(trainReg, kernelCfg)
            (x: Array[Array[Double]]) => f.predict(x)
          }, testX, trueMean))
      }
    }

    results.close()
    println(s"\nResults written to $DATA_DIR/bench_results.csv")
    println(s"Datasets: ${N_TRAINS.map(n => s"train_$n.csv").mkString(", ")}, test.csv")
  }
}

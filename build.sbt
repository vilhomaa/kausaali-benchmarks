ThisBuild / version      := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := "3.3.5"

// Source dependency on the sibling library checkout. The two repos are expected to
// live next to each other:
//
//   ~/workspace/omat/kausaali
//   ~/workspace/omat/kausaali-benchmarks
//
// If you prefer a binary dependency instead, run `sbt publishLocal` in the library
// and replace the `.dependsOn(...)` below with:
//   libraryDependencies += "io.github.vilhomaa" %% "kausaali" % "0.1.0-SNAPSHOT"
lazy val causalMLScala = ProjectRef(file("../kausaali"), "root")

lazy val benchmarks = (project in file("."))
  .dependsOn(causalMLScala)
  .settings(
    name           := "kausaali-benchmarks",
    publish / skip := true,
    libraryDependencies += "org.scala-lang.modules" %% "scala-parallel-collections" % "1.2.0"
  )

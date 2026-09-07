ThisBuild / version      := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := "3.3.5"

// Binary dependency on the published library (Maven Central).
//
// To develop against a local checkout instead, run `sbt publishLocal` in
// ../kausaali and point the version at the snapshot it prints, or switch back
// to a source dependency:
//
//   lazy val kausaali = ProjectRef(file("../kausaali"), "root")
//   lazy val benchmarks = (project in file(".")).dependsOn(kausaali)...

lazy val benchmarks = (project in file("."))
  .settings(
    name           := "kausaali-benchmarks",
    publish / skip := true,
    libraryDependencies ++= Seq(
      "io.github.vilhomaa"     %% "kausaali"                   % "0.1.1",
      "org.scala-lang.modules" %% "scala-parallel-collections" % "1.2.0"
    )
  )

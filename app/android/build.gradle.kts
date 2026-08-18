allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
}

// Forces every plugin module (including third-party ones like onnxruntime, which currently
// ships with an internally hardcoded compileSdk 33 that's too low for its own transitive
// androidx dependencies) to compile against the same, newer SDK as the app. Without this,
// Gradle's AAR-metadata check fails on mismatched compileSdk versions between the app and a
// plugin's own build config -- a known issue with plugins that haven't bumped compileSdk yet.
subprojects {
    // Skip :app -- it already sets its own compileSdk correctly (flutter.compileSdkVersion) and
    // by this point in evaluation order it's already fully evaluated, so calling afterEvaluate
    // on it here throws "Cannot run Project.afterEvaluate(Action) when the project is already
    // evaluated." Only the plugin modules (onnxruntime, etc.) need the override.
    if (project.name != "app") {
        afterEvaluate {
            extensions.findByType(com.android.build.gradle.BaseExtension::class.java)?.let { android ->
                android.compileSdkVersion(36)
            }
        }
    }
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}

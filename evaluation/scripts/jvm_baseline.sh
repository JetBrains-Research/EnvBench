#!/bin/bash

echo "Running JVM baseline script..."

set +e

# Source SDKMAN
export SDKMAN_DIR="/root/.sdkman"
source "${SDKMAN_DIR}/bin/sdkman-init.sh"

# Determine the build tool
if [[ -f "pom.xml" ]]; then
    BUILD_TOOL="maven"
    echo "Detected Maven project."
elif [[ -f "build.gradle" ]] || [[ -f "build.gradle.kts" ]]; then
    BUILD_TOOL="gradle"
    echo "Detected Gradle project."
else
    echo "Unsupported project type: No pom.xml or build.gradle(.kts) file found."
    exit 1
fi

# Function to extract Java version
extract_java_version() {
    if [[ "$BUILD_TOOL" == "maven" ]]; then
        # Attempt to extract from maven.compiler.source property
        JAVA_VERSION=$(xmlstarlet sel -N ns="http://maven.apache.org/POM/4.0.0" -t -v "/ns:project/ns:properties/ns:maven.compiler.source" pom.xml 2>/dev/null)
        if [[ -z "$JAVA_VERSION" ]]; then
            # Attempt to extract from maven-compiler-plugin configuration
            JAVA_VERSION=$(xmlstarlet sel -N ns="http://maven.apache.org/POM/4.0.0" \
            -t -v "/ns:project/ns:build/ns:plugins/ns:plugin[ns:artifactId='maven-compiler-plugin']/ns:configuration/ns:source" pom.xml 2>/dev/null)
        fi
        if [[ -z "$JAVA_VERSION" ]]; then
            # Attempt to extract from maven-enforcer-plugin RequireJavaVersion
            JAVA_VERSION=$(xmlstarlet sel -N ns="http://maven.apache.org/POM/4.0.0" \
            -t -v "/ns:project/ns:build/ns:plugins/ns:plugin[ns:artifactId='maven-enforcer-plugin']/ns:configuration/ns:rules/ns:requireJavaVersion/ns:version" pom.xml 2>/dev/null)
        fi
        echo "JAVA_VERSION: $JAVA_VERSION"
    elif [[ "$BUILD_TOOL" == "gradle" ]]; then
        # Attempt to extract from sourceCompatibility in build.gradle or build.gradle.kts
        JAVA_VERSION=$(grep -E 'sourceCompatibility\s*=\s*["'"'"']?[0-9.]+' build.gradle build.gradle.kts 2>/dev/null | \
        head -n1 | sed -E 's/.*sourceCompatibility\s*=\s*["'"'"']?([0-9.]+)["'"'"']?.*/\1/')
        if [[ -z "$JAVA_VERSION" ]]; then
            # Attempt to extract from java.sourceCompatibility in Kotlin DSL
            JAVA_VERSION=$(grep -E 'java\.sourceCompatibility\s*=\s*JavaVersion\.[A-Z0-9_]+' build.gradle.kts 2>/dev/null | \
            head -n1 | sed -E 's/.*JavaVersion\.([A-Z0-9_]+).*/\1/')
            # Map to Java version numbers
            case "$JAVA_VERSION" in
                VERSION_1_8) JAVA_VERSION="8" ;;
                VERSION_11) JAVA_VERSION="11" ;;
                VERSION_17) JAVA_VERSION="17" ;;
                VERSION_20) JAVA_VERSION="20" ;;
                *) JAVA_VERSION="" ;;
            esac
        fi
    fi

    # Map Java version to SDKMAN identifier
    if [[ -n "$JAVA_VERSION" ]]; then
        # Handle versions like 1.8 -> 8
        if [[ "$JAVA_VERSION" == 1.* ]]; then
            JAVA_VERSION=${JAVA_VERSION#1.}
        fi

        # Prepare the Java version for SDKMAN (e.g., 8 -> 8.0.382-tem)
        case "$JAVA_VERSION" in
            8) JAVA_VERSION="8.0.382-tem" ;;
            11) JAVA_VERSION="11.0.20-tem" ;;
            17) JAVA_VERSION="17.0.8-tem" ;;
            20) JAVA_VERSION="20.0.2-tem" ;;
            *) JAVA_VERSION="${JAVA_VERSION}.0-tem" ;;
        esac
    else
        echo "Java version not specified in project files."
        JAVA_VERSION=""
    fi
}

# Function to extract Maven version
extract_maven_version() {
    MAVEN_VERSION=$(xmlstarlet sel -t -v "/project/prerequisites/maven" pom.xml 2>/dev/null)
    if [[ -z "$MAVEN_VERSION" ]]; then
        # Attempt to extract from Maven Enforcer Plugin
        MAVEN_VERSION=$(xmlstarlet sel \
        -t -v "/project/build/plugins/plugin[artifactId='maven-enforcer-plugin']/configuration/rules/requireMavenVersion/version" pom.xml 2>/dev/null)
    fi

    if [[ -z "$MAVEN_VERSION" ]]; then
        echo "Maven version not specified."
        MAVEN_VERSION=""
    else
        echo "Maven version: $MAVEN_VERSION"
    fi
}

# Function to extract Gradle version
extract_gradle_version() {
    if [[ -f "gradle/wrapper/gradle-wrapper.properties" ]]; then
        GRADLE_VERSION=$(grep "distributionUrl" gradle/wrapper/gradle-wrapper.properties | \
        sed 's/.*gradle-\(.*\)-bin.zip.*/\1/')
    else
        echo "Gradle wrapper not found."
        GRADLE_VERSION=""
    fi
}

# Install and use the required Java version
extract_java_version
if [[ -n "$JAVA_VERSION" ]]; then
    echo "Installing and using Java $JAVA_VERSION with SDKMAN..."
    sdk install java "$JAVA_VERSION" && sdk use java "$JAVA_VERSION"
    if [[ $? -ne 0 ]]; then
        echo "Failed to install or use Java version $JAVA_VERSION"
    fi
else
    echo "Using default Java version."
fi

# Install and use the required build tool
if [[ "$BUILD_TOOL" == "maven" ]]; then
    extract_maven_version
    if [[ -n "$MAVEN_VERSION" ]]; then
        echo "Installing and using Maven $MAVEN_VERSION with SDKMAN..."
        sdk install maven "$MAVEN_VERSION" && sdk use maven "$MAVEN_VERSION"
        if [[ $? -ne 0 ]]; then
            echo "Failed to install or use Maven version $MAVEN_VERSION"
        fi
    else
        echo "Using default Maven version."
    fi
    # # Build the project with Maven
    # if [[ -f "mvnw" ]]; then
    #     echo "Using Maven Wrapper."
    #     chmod +x mvnw
    #     ./mvnw clean install
    # else
    #     mvn clean install
    # fi
elif [[ "$BUILD_TOOL" == "gradle" ]]; then
    extract_gradle_version
    if [[ -n "$GRADLE_VERSION" ]]; then
        echo "Installing and using Gradle $GRADLE_VERSION with SDKMAN..."
        sdk install gradle "$GRADLE_VERSION" && sdk use gradle "$GRADLE_VERSION"
        if [[ $? -ne 0 ]]; then
            echo "Failed to install or use Gradle version $GRADLE_VERSION"
        fi
    else
        echo "Using default Gradle version."
    fi
    # # Build the project with Gradle
    # if [[ -f "gradlew" ]]; then
    #     echo "Using Gradle Wrapper."
    #     chmod +x gradlew
    #     ./gradlew clean build
    # else
    #     gradle clean build
    # fi
else
    echo "Unsupported build tool: $BUILD_TOOL"
fi

set -e

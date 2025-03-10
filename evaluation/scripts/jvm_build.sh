#!/bin/bash

set -e

# Create output directory
mkdir -p build_output
chmod -R 777 .

# initial contents
printf '{}\n' > build_output/results.json  

# If a bootstrap script exists, run it first
if [ -f "./bootstrap_script.sh" ]; then
  echo "Running bootstrap script..."
  source ./bootstrap_script.sh
fi

echo "Running JVM build script..."

# Install jq
apt-get update -yqq && apt-get install -yqq jq
# Initialize build issues count and diagnostic log
issues="-1"
diagnostic_log="[]"

# Detect build tool and run appropriate build command
if [ -f "gradlew" ] || [ -f "./gradlew" ]; then
    echo "Gradle project detected"
    chmod +x ./gradlew
    ./gradlew build -x test --console=plain > build.log 2>&1 || true
    issues=$(grep -c "FAILURE" build.log || true)
    build_tool="gradle"
    # Extract failure messages into diagnostic log
    diagnostic_log=$(grep "FAILURE" build.log | jq -R -s 'split("\n")[:-1]' || echo "[]")
elif [ -f "build.gradle" ] || [ -f "build.gradle.kts" ]; then
    echo "Gradle project detected"
    gradle build -x test --console=plain > build.log 2>&1 || true
    issues=$(grep -c "FAILURE" build.log || true)
    build_tool="gradle"
    # Extract failure messages into diagnostic log
    diagnostic_log=$(grep "FAILURE" build.log | jq -R -s 'split("\n")[:-1]' || echo "[]")
elif [ -f "mvnw" ] || [ -f "./mvnw" ]; then
    echo "Maven project detected"
    chmod +x ./mvnw
    ./mvnw compile -DskipTests -B > build.log 2>&1 || true
    issues=$(grep -c "\[ERROR\]" build.log || true)
    build_tool="maven"
    # Extract error messages into diagnostic log
    diagnostic_log=$(grep "\[ERROR\]" build.log | jq -R -s 'split("\n")[:-1]' || echo "[]")
elif [ -f "pom.xml" ]; then
    echo "Maven project detected"
    mvn compile -DskipTests -B > build.log 2>&1 || true
    issues=$(grep -c "\[ERROR\]" build.log || true)
    build_tool="maven"
    # Extract error messages into diagnostic log
    diagnostic_log=$(grep "\[ERROR\]" build.log | jq -R -s 'split("\n")[:-1]' || echo "[]")
else
    echo "No supported build tool found"
    exit 1
fi

# Add build tool and issues to results
jq --arg tool "$build_tool" \
   --arg issues "$issues" \
   '. + {"build_tool": $tool, "issues_count": ($issues|tonumber)}' \
    build_output/results.json > build_output/temp.json && \
    mv build_output/temp.json build_output/results.json

# Add diagnostic log to results
echo "$diagnostic_log" > build_output/diagnostic_log.json
jq -s '.[0] * {"diagnostic_log": .[1]}' build_output/results.json build_output/diagnostic_log.json > build_output/temp.json && \
    mv build_output/temp.json build_output/results.json

chmod -R 777 .
exit 0

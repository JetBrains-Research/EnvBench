"""Commands for collecting repository context for environment setup."""

PYTHON_CONTEXT_COMMANDS = [
    # Repository structure
    'tree -a -L 3 || ls -R',  # Fallback to ls -R if tree not available
    
    # Core setup files - direct read
    'for f in setup.py pyproject.toml setup.cfg tox.ini; do '
    'if [ -f "$f" ]; then echo -e "\\n=== $f ==="; cat "$f"; fi; done',
    
    # Requirements files - direct read
    'for f in requirements.txt requirements/*.txt; do '
    'if [ -f "$f" ]; then echo -e "\\n=== $f ==="; cat "$f"; fi; done',
    
    # Documentation - direct read
    'for f in README.md INSTALL.md SETUP.md docs/INSTALL.md docs/SETUP.md; do '
    'if [ -f "$f" ]; then echo -e "\\n=== $f ==="; cat "$f"; fi; done',
    
    # Find and show all Python build files
    'find . -type f \\( '
    '-name "requirements*.txt" -o -name "setup.py" -o -name "pyproject.toml" -o -name "setup.cfg" '
    '\\) | while read f; do echo -e "\\n=== $f ==="; cat "$f"; done',
    
    # Python version info
    'find . -type f -name "*.py" -exec grep -l "python_version\|python_requires" {} \\;',
    
    # Environment files
    'find . -type f \\( -name ".env*" -o -name "*.env" -o -name "Dockerfile*" \\) | '
    'while read f; do echo -e "\\n=== $f ==="; cat "$f"; done',
    
    # Docker files - direct read
    'for f in Dockerfile docker-compose.yml docker-compose.yaml; do '
    'if [ -f "$f" ]; then echo -e "\\n=== $f ==="; cat "$f"; fi; done',
    
    # Python setup instructions in docs
    'find . -type f -name "*.md" -exec grep -i "python\|pip\|requirements\|virtualenv\|venv" {} \\;',
    
    # Additional Python files that might contain dependencies
    'find . -maxdepth 3 -type f -name "__init__.py" | '
    'while read f; do echo -e "\\n=== $f ==="; cat "$f"; done'
]

JVM_CONTEXT_COMMANDS = [
    # Repository structure
    'tree -a -L 3 || ls -R',  # Fallback to ls -R if tree not available
    
    # Core build files - direct read
    'for f in pom.xml build.gradle settings.gradle gradle.properties gradlew.bat gradlew; do '
    'if [ -f "$f" ]; then echo -e "\\n=== $f ==="; cat "$f"; fi; done',
    
    # Documentation - direct read
    'for f in README.md INSTALL.md SETUP.md docs/INSTALL.md docs/SETUP.md; do '
    'if [ -f "$f" ]; then echo -e "\\n=== $f ==="; cat "$f"; fi; done',
    
    # Find and show all JVM build files
    'find . -type f \\( '
    '-name "pom.xml" -o -name "build.gradle*" -o -name "settings.gradle*" -o -name "gradle.properties" '
    '\\) | while read f; do echo -e "\\n=== $f ==="; cat "$f"; done',
    
    # Java version info
    'find . -type f \\( -name "*.java" -o -name "*.gradle" \\) -exec grep -l '
    '"sourceCompatibility\|targetCompatibility\|java.version" {} \\;',
    
    # Environment files
    'find . -type f \\( -name ".env*" -o -name "*.env" -o -name "Dockerfile*" \\) | '
    'while read f; do echo -e "\\n=== $f ==="; cat "$f"; done',
    
    # Docker files - direct read
    'for f in Dockerfile docker-compose.yml docker-compose.yaml; do '
    'if [ -f "$f" ]; then echo -e "\\n=== $f ==="; cat "$f"; fi; done',
    
    # Java setup instructions in docs
    'find . -type f -name "*.md" -exec grep -i "java\|jdk\|maven\|gradle\|build" {} \\;',
    
    # Additional build files that might be important
    'find . -maxdepth 3 -type f -name "*.gradle" | '
    'while read f; do echo -e "\\n=== $f ==="; cat "$f"; done',
    
    # Maven wrapper and Gradle wrapper configs
    'for f in .mvn/wrapper/maven-wrapper.properties gradle/wrapper/gradle-wrapper.properties; do '
    'if [ -f "$f" ]; then echo -e "\\n=== $f ==="; cat "$f"; fi; done'
] 

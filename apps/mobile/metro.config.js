const { getDefaultConfig } = require("expo/metro-config");
const path = require("path");

const monorepoRoot = path.resolve(__dirname, "../..");

const config = getDefaultConfig(__dirname);

// Watch the entire monorepo so Metro sees changes to workspace packages
config.watchFolders = [monorepoRoot];

// Tell Metro where to find node_modules (hoisted to root by .npmrc)
config.resolver.nodeModulesPaths = [
  path.resolve(__dirname, "node_modules"),
  path.resolve(monorepoRoot, "node_modules"),
];

// Prevent Metro from finding duplicate React/RN copies in nested node_modules
config.resolver.disableHierarchicalLookup = true;

// Enable package.json "exports" field resolution and prefer react-native/browser builds
config.resolver.unstable_enablePackageExports = true;
config.resolver.unstable_conditionNames = ["react-native", "browser", "require"];

module.exports = config;

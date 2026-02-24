import { View, Text, StyleSheet } from "react-native";
import { WaveForecastClient } from "@wave-forecast/api-client";

export default function HomeScreen() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Wave Forecast</Text>
      <Text style={styles.subtitle}>Build working.</Text>
      <Text style={styles.detail}>
        API client loaded: {WaveForecastClient ? "yes" : "no"}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#0F172A",
  },
  title: {
    fontSize: 28,
    fontWeight: "700",
    color: "#F8FAFC",
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 18,
    color: "#94A3B8",
    marginBottom: 16,
  },
  detail: {
    fontSize: 14,
    color: "#64748B",
  },
});

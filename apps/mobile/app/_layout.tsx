import { Stack } from "expo-router";
import { StatusBar } from "expo-status-bar";

export default function RootLayout() {
  return (
    <>
      <StatusBar style="light" />
      <Stack
        screenOptions={{
          headerStyle: { backgroundColor: "#0F172A" },
          headerTintColor: "#F8FAFC",
          contentStyle: { backgroundColor: "#0F172A" },
        }}
      >
        <Stack.Screen
          name="index"
          options={{ title: "Wave Forecast" }}
        />
      </Stack>
    </>
  );
}

package com.kiseki.momentwriter

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val settingsRepo = SettingsRepository(applicationContext)

        setContent {
            MomentWriterTheme {
                val navController = rememberNavController()
                NavHost(navController, startDestination = "editor") {
                    composable("editor") {
                        EditorScreen(
                            onNavigateToSettings = { navController.navigate("settings") },
                            settingsRepo = settingsRepo
                        )
                    }
                    composable("settings") {
                        SettingsScreen(
                            onBack = { navController.popBackStack() },
                            settingsRepo = settingsRepo
                        )
                    }
                }
            }
        }
    }
}

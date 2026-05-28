package com.kiseki.momentwriter

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.dataStore: DataStore<Preferences> by preferencesDataStore("settings")

class SettingsRepository(private val context: Context) {

    private object Keys {
        val TOKEN = stringPreferencesKey("token")
        val OWNER = stringPreferencesKey("owner")
        val REPO = stringPreferencesKey("repo")
        val DEFAULT_LOCATION = stringPreferencesKey("default_location")
    }

    val settingsFlow: Flow<AppSettings> = context.dataStore.data.map { prefs ->
        AppSettings(
            token = prefs[Keys.TOKEN] ?: "",
            owner = prefs[Keys.OWNER] ?: "Ki-Seki",
            repo = prefs[Keys.REPO] ?: "ki-seki.github.io",
            defaultLocation = prefs[Keys.DEFAULT_LOCATION] ?: "Beijing"
        )
    }

    suspend fun save(settings: AppSettings) {
        context.dataStore.edit { prefs ->
            prefs[Keys.TOKEN] = settings.token
            prefs[Keys.OWNER] = settings.owner
            prefs[Keys.REPO] = settings.repo
            prefs[Keys.DEFAULT_LOCATION] = settings.defaultLocation
        }
    }
}

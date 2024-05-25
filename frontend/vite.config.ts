import { resolve } from 'path'
import { defineConfig, UserConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig(({ command }) => {
	const config: UserConfig = {
		plugins: [react()],
		resolve: {
			alias: {
				'@': resolve(__dirname, './src'),
			},
		},
	}

	if (command !== 'serve') {
		config.base = '/red-hack-time-series-analyzer/'
	}

	return config
})

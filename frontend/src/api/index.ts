import Axios, { AxiosResponse } from 'axios'

const baseURL = import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:5000'
// 'https://326d-2601-647-5800-704-3d07-56ce-7d62-658b.ngrok-free.app/api'

export const axios = Axios.create({
	baseURL,
})

export type Series = 'RESPONSE' | 'APDEX' | 'THROUGHPUT'
export type Point = {
	point: string
	value: number
}

export const getSeries = (series: Series): Promise<AxiosResponse<Point[]>> => {
	return axios.get<Point[]>('/get-series', {
		params: {
			'time-series': series,
		},
	})
}

export default { getSeries }

import Axios, { AxiosResponse } from 'axios'

const baseURL =
	'https://time-series-outliers-detector-7d3cc62cf333.herokuapp.com'
// 'https://326d-2601-647-5800-704-3d07-56ce-7d62-658b.ngrok-free.app/api'

export const axios = Axios.create({
	baseURL,
})

export type Metric = 'RESPONSE' | 'APDEX' | 'THROUGHPUT' | 'ERROR'
export type Point = {
	point: string
	value: number
}

export const getSeries = (series: Metric): Promise<AxiosResponse<Point[]>> => {
	return axios.get<Point[]>('/get-series', {
		params: {
			'time-series': series,
		},
	})
}

export const getAnomalies = ({
	startTime,
	endTime,
	timeSeries,
}: {
	startTime: string
	endTime: string
	timeSeries: Metric
}) => {
	return axios.get<string[]>('/get-anomilies', {
		params: {
			'time-series': timeSeries,
			start: startTime,
			end: endTime,
		},
	})
}

export default { getSeries }

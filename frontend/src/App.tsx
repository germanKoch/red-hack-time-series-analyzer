import { useEffect, useState } from 'react'

import api, { Point } from '@/api'

import { Chart } from './components/ChartArm'

import './App.css'

function App() {
	const [responseSeries, setResponseSeries] = useState<Point[]>([])

	useEffect(() => {
		const fetchData = async () => {
			const { data } = await api.getSeries('RESPONSE')
			setResponseSeries(data)
		}

		fetchData()
	}, [])
	return <Chart data={responseSeries.slice(100, 3000)} />
}

export default App

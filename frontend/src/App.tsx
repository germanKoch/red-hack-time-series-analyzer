import { useCallback, useEffect, useState } from 'react'

import dayjs from 'dayjs'
import ArrowBackIosIcon from '@mui/icons-material/ArrowBackIos'
import ArrowForwardIosIcon from '@mui/icons-material/ArrowForwardIos'
import {
	Box,
	Button,
	CircularProgress,
	IconButton,
	MenuItem,
	Select,
	SelectProps,
	Typography,
} from '@mui/material'
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs'
import {
	DateTimePicker,
	DateTimePickerProps,
} from '@mui/x-date-pickers/DateTimePicker'
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider'

import api, { getAnomalies, Metric, Point } from '@/api'

import { Chart } from './components/ChartArm'

import './App.css'

const PageSize = 1000

function App() {
	const [isLoading, setIsLoading] = useState<boolean>(false)
	const [responseSeries, setResponseSeries] = useState<Point[]>([])
	const [apdexSeries, setApdexSeries] = useState<Point[]>([])
	const [throughputSeries, setThroughputSeries] = useState<Point[]>([])
	const [errorSeries, setErrorSeries] = useState<Point[]>([])

	const [series, setSeries] = useState<Point[]>([])
	const [anomalies, setAnomalies] = useState<string[]>([])
	const [page, setPage] = useState<number>(0)
	const [metric, setMetric] = useState<Metric>('RESPONSE')

	const [startTime, setStartTime] = useState<dayjs.Dayjs | null>(
		dayjs('2024-04-15'),
	)
	const [endTime, setEndTime] = useState<dayjs.Dayjs | null>(
		dayjs('2024-04-16'),
	)

	const handleStartTimeChange: DateTimePickerProps<dayjs.Dayjs>['onChange'] = (
		value,
	) => {
		setStartTime(value)
	}

	const handleEndTimeChange: DateTimePickerProps<dayjs.Dayjs>['onChange'] = (
		value,
	) => {
		setEndTime(value)
	}

	const handleMetricChange: SelectProps['onChange'] = (e) => {
		setMetric(e.target.value as Metric)
	}

	const filterSeriesByDates = useCallback(
		(s: Point[]) => {
			return s.filter(({ point }) => {
				const pointDate = new Date(point)
				if (startTime && endTime) {
					return startTime.isBefore(pointDate) && endTime.isAfter(pointDate)
				} else {
					return false
				}
			})
		},
		[endTime, startTime],
	)

	const getSeriesData = (m: Metric) => {
		switch (m) {
			case 'RESPONSE':
				return responseSeries
			case 'THROUGHPUT':
				return throughputSeries
			case 'APDEX':
				return apdexSeries
			case 'ERROR':
				return errorSeries
			default:
				return responseSeries
		}
	}

	const handleGetAnomalies: React.FormEventHandler = async (e) => {
		e.preventDefault()
		setIsLoading(true)

		const newSeries = getSeriesData(metric)

		setSeries(filterSeriesByDates(newSeries))

		const anomalies = (
			await getAnomalies({
				startTime: startTime?.format('YYYY-MM-DDTHH:mm:ss') || '',
				endTime: endTime?.format('YYYY-MM-DDTHH:mm:ss') || '',
				timeSeries: metric,
			})
		).data

		setAnomalies(anomalies)
		setIsLoading(false)
	}

	useEffect(() => {
		const fetchData = async () => {
			setIsLoading(true)
			const [responseData, throughputData, apdexData, errorData] =
				await Promise.all([
					api.getSeries('RESPONSE'),
					api.getSeries('THROUGHPUT'),
					api.getSeries('APDEX'),
					api.getSeries('ERROR'),
				])
			const anomalies = (
				await getAnomalies({
					startTime: startTime?.format('YYYY-MM-DDTHH:mm:ss') || '',
					endTime: endTime?.format('YYYY-MM-DDTHH:mm:ss') || '',
					timeSeries: metric,
				})
			).data

			setResponseSeries(responseData.data)
			setThroughputSeries(throughputData.data)
			setApdexSeries(apdexData.data)
			setErrorSeries(errorData.data)

			setAnomalies(anomalies)

			setSeries(filterSeriesByDates(responseData.data))
			setIsLoading(false)
		}

		fetchData()
	}, [])
	return (
		<LocalizationProvider dateAdapter={AdapterDayjs}>
			<div>
				<form className="form" onSubmit={handleGetAnomalies}>
					<DateTimePicker
						minDateTime={dayjs('2024-04-15')}
						maxDateTime={dayjs('2024-05-17')}
						value={startTime}
						onChange={handleStartTimeChange}
					/>
					<DateTimePicker
						minDateTime={dayjs('2024-04-15')}
						maxDateTime={dayjs('2024-05-17')}
						value={endTime}
						onChange={handleEndTimeChange}
					/>

					<Select<Metric>
						value={metric}
						label="Metric"
						onChange={handleMetricChange}
					>
						<MenuItem value={'RESPONSE'}>Response</MenuItem>
						<MenuItem value={'APDEX'}>Apdex</MenuItem>
						<MenuItem value={'THROUGHPUT'}>Throughput</MenuItem>
						<MenuItem value={'ERROR'}>Error</MenuItem>
					</Select>

					<Button type="submit" variant="contained">
						Get anomalies
					</Button>
				</form>
				<div className="toolbar">
					<IconButton onClick={() => setPage(page - 1)} disabled={page <= 0}>
						<ArrowBackIosIcon />
					</IconButton>
					<Typography fontSize={20}>Pagination</Typography>
					<IconButton
						onClick={() => setPage(page + 1)}
						disabled={(page + 1) * PageSize >= series.length}
					>
						<ArrowForwardIosIcon />
					</IconButton>
				</div>
				{isLoading ? (
					<Box
						sx={{
							display: 'flex',
							width: '100%',
							justifyContent: 'center',
							margin: '30px',
						}}
					>
						<CircularProgress />
					</Box>
				) : (
					<Chart
						data={series.slice(page * PageSize, (page + 1) * PageSize)}
						anomalies={anomalies}
					/>
				)}
			</div>
		</LocalizationProvider>
	)
}

export default App

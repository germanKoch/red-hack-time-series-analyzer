import { useEffect, useState } from 'react'

import dayjs from 'dayjs'
import ArrowBackIosIcon from '@mui/icons-material/ArrowBackIos'
import ArrowForwardIosIcon from '@mui/icons-material/ArrowForwardIos'
import { Button, IconButton, Typography } from '@mui/material'
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs'
import {
	DateTimePicker,
	DateTimePickerProps,
} from '@mui/x-date-pickers/DateTimePicker'
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider'

import api, { Point } from '@/api'

import { Chart } from './components/ChartArm'

import './App.css'

const PageSize = 1000

function App() {
	const [responseSeries, setResponseSeries] = useState<Point[]>([])
	const [series, setSeries] = useState<Point[]>([])
	const [page, setPage] = useState<number>(0)

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

	const handleGetAnomalies: React.FormEventHandler = (e) => {
		e.preventDefault()

		console.log(startTime, endTime)

		setSeries(
			responseSeries.filter(({ point }) => {
				const pointDate = new Date(point)
				if (startTime && endTime) {
					return startTime.isBefore(pointDate) && endTime.isAfter(pointDate)
				} else {
					return false
				}
			}),
		)
	}

	console.log(series)

	useEffect(() => {
		const fetchData = async () => {
			const { data } = await api.getSeries('RESPONSE')
			setResponseSeries(data)
			setSeries(data)
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
				<Chart data={series.slice(page * PageSize, (page + 1) * PageSize)} />
			</div>
		</LocalizationProvider>
	)
}

export default App

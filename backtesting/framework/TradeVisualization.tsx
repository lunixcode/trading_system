import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceDot, Legend, ResponsiveContainer } from 'recharts';

interface TradeData {
  timestamp: string;
  price: number;
  volume: number;
  entry?: boolean;
  exit?: boolean;
  tradeType?: 'buy' | 'sell';
}

interface TradeVisualizationProps {
  data: TradeData[];
  title?: string;
}

const TradeVisualization: React.FC<TradeVisualizationProps> = ({ data, title }) => {
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-4 border border-gray-200 shadow-lg rounded">
          <p className="text-sm">{`Time: ${label}`}</p>
          <p className="text-sm text-blue-600">{`Price: $${payload[0].value}`}</p>
          {payload[0].payload.entry && (
            <p className="text-sm text-green-600">Entry Point</p>
          )}
          {payload[0].payload.exit && (
            <p className="text-sm text-red-600">Exit Point</p>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full h-96 p-4">
      {title && <h2 className="text-xl font-semibold mb-4">{title}</h2>}
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={data}
          margin={{
            top: 20,
            right: 30,
            left: 20,
            bottom: 10
          }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="timestamp" 
            tick={{ fontSize: 12 }} 
            tickFormatter={(value) => value.split(' ')[1]}
          />
          <YAxis 
            domain={['dataMin - 1', 'dataMax + 1']}
            tick={{ fontSize: 12 }}
          />
          <Tooltip content={<CustomTooltip active={undefined} payload={undefined} label={undefined} />} />
          <Legend />
          
          {/* Price Line */}
          <Line
            type="monotone"
            dataKey="price"
            stroke="#2196F3"
            strokeWidth={2}
            dot={false}
            name="Price"
          />

          {/* Entry Points */}
          {data.map((point, index) => 
            point.entry && (
              <ReferenceDot
                key={`entry-${index}`}
                x={point.timestamp}
                y={point.price}
                r={6}
                fill="#4CAF50"
                stroke="none"
              />
            )
          )}

          {/* Exit Points */}
          {data.map((point, index) => 
            point.exit && (
              <ReferenceDot
                key={`exit-${index}`}
                x={point.timestamp}
                y={point.price}
                r={6}
                fill="#F44336"
                stroke="none"
              />
            )
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default TradeVisualization;
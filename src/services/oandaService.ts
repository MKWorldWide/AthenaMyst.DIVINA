import axios, { AxiosInstance } from 'axios';
import { oandaConfig, tradingSettings } from '../config/oanda';

export interface OandaOrder {
  order: {
    units: string;
    instrument: string;
    timeInForce: string;
    type: string;
    positionFill: string;
    stopLossOnFill?: {
      price: string;
      timeInForce: string;
    };
    takeProfitOnFill?: {
      price: string;
    };
  };
}

export interface OandaTrade {
  id: string;
  instrument: string;
  price: string;
  currentUnits: string;
  initialUnits: string;
  state: string;
  openTime: string;
  stopLossOrder?: {
    price: string;
  };
  takeProfitOrder?: {
    price: string;
  };
}

export class OandaService {
  private client: AxiosInstance;
  private streamClient: AxiosInstance;
  private accountId: string;

  constructor() {
    this.accountId = oandaConfig.accountId;
    
    this.client = axios.create({
      baseURL: oandaConfig.baseUrl,
      headers: {
        'Authorization': `Bearer ${oandaConfig.apiKey}`,
        'Content-Type': 'application/json',
      },
    });

    this.streamClient = axios.create({
      baseURL: oandaConfig.streamUrl,
      headers: {
        'Authorization': `Bearer ${oandaConfig.apiKey}`,
        'Content-Type': 'application/json',
      },
      responseType: 'stream',
    });
  }

  private formatPrice(price: number): string {
    return price.toFixed(5);
  }

  async getAccountSummary() {
    try {
      const response = await this.client.get(`/accounts/${this.accountId}/summary`);
      return response.data.account;
    } catch (error) {
      console.error('Error fetching account summary:', error);
      throw error;
    }
  }

  async getCurrentPrice(instrument: string) {
    try {
      const response = await this.client.get(`/accounts/${this.accountId}/pricing?instruments=${instrument}`);
      const priceData = response.data.prices[0];
      return {
        bid: parseFloat(priceData.bids[0].price),
        ask: parseFloat(priceData.asks[0].price),
        time: priceData.time
      };
    } catch (error) {
      console.error('Error fetching current price:', error);
      throw error;
    }
  }

  async placeOrder(orderData: OandaOrder) {
    try {
      const response = await this.client.post(
        `/accounts/${this.accountId}/orders`,
        orderData
      );
      return response.data.orderCreateTransaction;
    } catch (error: any) {
      console.error('Error placing order:', error.response?.data || error.message);
      throw new Error(`Failed to place order: ${error.response?.data?.errorMessage || error.message}`);
    }
  }

  async getOpenTrades(): Promise<OandaTrade[]> {
    try {
      const response = await this.client.get(`/accounts/${this.accountId}/openTrades`);
      return response.data.trades;
    } catch (error: any) {
      console.error('Error fetching open trades:', error.response?.data || error.message);
      throw new Error(`Failed to fetch open trades: ${error.response?.data?.errorMessage || error.message}`);
    }
  }

  async closeTrade(tradeId: string): Promise<{ success: boolean; message: string }> {
    try {
      await this.client.put(`/accounts/${this.accountId}/trades/${tradeId}/close`);
      return {
        success: true,
        message: `Trade ${tradeId} closed successfully`,
      };
    } catch (error: any) {
      console.error('Error closing trade:', error.response?.data || error.message);
      return {
        success: false,
        message: `Failed to close trade: ${error.response?.data?.errorMessage || error.message}`,
      };
    }
  }

  async getCandles(instrument: string, granularity: string = 'M5', count: number = 50): Promise<any> {
    try {
      const response = await this.client.get(
        `/instruments/${instrument}/candles`,
        { 
          params: {
            granularity,
            count,
            price: 'M'  // Midpoint candles
          } 
        }
      );
      return response.data.candles;
    } catch (error: any) {
      console.error('Error fetching candles:', error.response?.data || error.message);
      throw new Error(`Failed to fetch candles: ${error.response?.data?.errorMessage || error.message}`);
    }
  }

  async getTradeDetails(tradeId: string): Promise<OandaTrade> {
    try {
      const response = await this.client.get(`/accounts/${this.accountId}/trades/${tradeId}`);
      return response.data.trade as OandaTrade;
    } catch (error: any) {
      console.error('Error fetching trade details:', error.response?.data || error.message);
      throw new Error(`Failed to fetch trade details: ${error.response?.data?.errorMessage || error.message}`);
    }
  }
}

// Export a singleton instance
export const oandaService = new OandaService();

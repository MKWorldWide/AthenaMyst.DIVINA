const request = require('supertest');
const app = require('../api/index');

describe('AthenaMyst API', () => {
  // Basic health check to ensure server responds
  it('returns healthy status', async () => {
    const res = await request(app).get('/health');
    expect(res.statusCode).toBe(200);
    expect(res.body.status).toBe('healthy');
  });

  // Ensure analytics endpoint accepts data and reports stats
  it('collects analytics data', async () => {
    const payload = { type: 'page_view', data: { sessionId: 'jest-session' } };
    const postRes = await request(app).post('/analytics').send(payload);
    expect(postRes.statusCode).toBe(200);
    expect(postRes.body.success).toBe(true);

    const statsRes = await request(app).get('/analytics');
    expect(statsRes.statusCode).toBe(200);
    expect(statsRes.body.totalEntries).toBeGreaterThan(0);
  });
});

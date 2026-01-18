class JobAnalytics:
    """
    Аналитика рынка вакансий Data Science
    """

    def __init__(self, df):
        self.df = df

    def top_locations(self, n=10):
        return self.df['Location'].value_counts().head(n)

    def top_jobs(self, n=10):
        return self.df['Job Title'].value_counts().head(n)

    def top_employers(self, n=10):
        return self.df['Company Name'].value_counts().head(n)

    def average_salary(self):
        return self.df['Average salary, $K'].astype(float).mean()

    def best_rated_companies(self, min_rating=4.0):
        return (
            self.df[['Company Name', 'Company Rating']]
            .drop_duplicates()
            .sort_values('Company Rating', ascending=False)
            .query('`Company Rating` >= @min_rating')
        )

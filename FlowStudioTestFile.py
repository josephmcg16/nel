from typing import Any
import pandas as pd
from attr import dataclass


@dataclass
class TestPoint:
    job_number: str
    test_number: str
    test_title: str
    test_description: str
    testpoint_number: int
    comment:str
    data: pd.DataFrame
    columns: list[str]
    columns_dict: list[dict]


class FlowStudioTestFile:
    def __init__(self, path: str):
        self.path = path
        print("Reading Excel file...")
        excel_sheets = pd.read_excel(path, sheet_name=None, header=None)

        first_sheet = excel_sheets["Test Data"]
        idx = first_sheet[first_sheet[0] == "Comment"].index
        self.comments = first_sheet.iloc[idx[0]+2:idx[1]-2][0].to_numpy()

        self.testpoints: dict[int, TestPoint] = self._get_testpoints(excel_sheets)
        print("Excel file read.")
    
    def _get_testpoints(self, excel_sheets: pd.DataFrame) -> dict:
        def _get_columns_dict(sheet_df, column_names):
            df = pd.DataFrame(
                sheet_df.iloc[8:11].to_numpy().T,
                columns=['ID', 'Description', 'Units']
            ).fillna(method='ffill')
            df['Name'] = column_names
            df['ID'] = df['ID'].str.replace("ID=", "").astype(int)
            return df.to_dict('records')
        testpoints = dict()
        for sheet_df in list(excel_sheets.values()):
            if sheet_df.iloc[0, 0] == "Test Point Scan Data":
                column_names = list(sheet_df.iloc[9].str.cat(
                    sheet_df.iloc[10], sep=', '))
                tp = TestPoint(
                    job_number=sheet_df.iloc[2, 1],
                    test_number=sheet_df.iloc[3, 1],
                    test_title=sheet_df.iloc[4, 1],
                    test_description=sheet_df.iloc[5, 1],
                    testpoint_number=sheet_df.iloc[6, 1],
                    comment=self.comments[sheet_df.iloc[6, 1]-1],
                    data=pd.DataFrame(
                        sheet_df.values[11:],
                        columns=column_names,
                    ),
                    columns=column_names,
                    columns_dict=_get_columns_dict(sheet_df, column_names)
                )
                testpoints[tp.testpoint_number] = tp
        return testpoints
    

    def __call__(self, testpoint_number: int) -> TestPoint:
        return self.testpoints[testpoint_number]

    def __get_item__(self, testpoint_number: int) -> TestPoint:
        return self.testpoints[testpoint_number]

    def __iter__(self):
        return iter(self.testpoints.values())
    
    def __len__(self):
        return len(self.testpoints)


if __name__ == "__main__":
    testfile = FlowStudioTestFile("epat_upgrade/data/FRIP23_T002502.xlsx")

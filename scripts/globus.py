import globus_sdk
from typing import List


source_uuid = {
   "QuijoteLHRusty": "ee0786de-a7cc-11ed-a2a6-8383522b48d9"
}


def download_one(number: int, tdata: globus_sdk.TransferData, what: List[str] = ['lin', 'params', 'nonlin']):
   number = str(number).zfill(4)

   source_linear_dis_path = f'/~/Linear/LH{number}/4/dis.npy'
   source_params_path = f'/~/Linear/LH{number}/4/params.npy'
   source_nonlinear_dis_path = f'/~/Nonlinear/LH{number}/4/dis.npy'

   dest_data_dir = f'/user_data/ajliang/Quijote'
   dest_linear_dis_path = f'{dest_data_dir}/LH{number}/lin.npy'
   dest_params_path = f'{dest_data_dir}/LH{number}/params.npy'
   dest_nonlinear_dis_path = f'{dest_data_dir}/LH{number}/nonlin.npy'

   if 'lin' in what:
      tdata.add_item(
         source_path=source_linear_dis_path,
         destination_path=dest_linear_dis_path,
      )
   if 'params' in what:
      tdata.add_item(
         source_path=source_params_path,
         destination_path=dest_params_path,
      )
   if 'nonlin' in what:
      tdata.add_item(
         source_path=source_nonlinear_dis_path,
         destination_path=dest_nonlinear_dis_path,
      )


def download_all(tc: globus_sdk.TransferClient, tdata: globus_sdk.TransferData):
   # for number in [1045, 1416, 715, 1460, 700, 727, 662, 482, 853, 1026, 699]:
   #    download_one(number=number, tdata=tdata, what=['lin', 'params', 'nonlin'])
   for number in [1045, 590, 1988, 785, 506]:
      download_one(number=number, tdata=tdata, what=['lin', 'params', 'nonlin'])

   # download_one(number=663, tdata=tdata, what=['lin', 'params', 'nonlin'])
   # for i in range(2000):
   #    download_one(number=i, tdata=tdata, what=[
   #       # 'lin',
   #       'params',
   #       # 'nonlin'
   #    ])


if __name__ == '__main__':

   source_endpoint_id = source_uuid['QuijoteLHRusty']
   destination_endpoint_id = '2d6428b8-58af-11ed-89db-ede5bae4f491'

   CLIENT_ID = 'b429e547-eebc-4b2d-8594-e52152f37e74'
   client = globus_sdk.NativeAppAuthClient(CLIENT_ID)
   client.oauth2_start_flow()
   authorize_url = client.oauth2_get_authorize_url()
   print(f"Please go to this URL and login:\n\n{authorize_url}\n")

   auth_code = input("Please enter the code you get after login here: ").strip()
   token_response = client.oauth2_exchange_code_for_tokens(auth_code)
   globus_auth_data = token_response.by_resource_server["auth.globus.org"]
   globus_transfer_data = token_response.by_resource_server["transfer.api.globus.org"]

   AUTH_TOKEN = globus_auth_data["access_token"]
   TRANSFER_TOKEN = globus_transfer_data["access_token"]

   authorizer = globus_sdk.AccessTokenAuthorizer(TRANSFER_TOKEN)
   tc = globus_sdk.TransferClient(authorizer=authorizer)
   tdata = globus_sdk.TransferData(tc, source_endpoint_id,
                                 destination_endpoint_id,
                                 label="Download OOD Quijote Simulations",
                                 sync_level="checksum")

   download_all(tc=tc, tdata=tdata)

   transfer_result = tc.submit_transfer(tdata)
   print("task_id =", transfer_result["task_id"])

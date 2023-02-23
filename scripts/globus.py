import globus_sdk


def download_one(number: int, tdata: globus_sdk.TransferData):
   number = str(number).zfill(4)

   source_linear_dis_path = f'/~/Linear/LH{number}/4/dis.npy'
   source_params_path = f'/~/Linear/LH{number}/4/params.npy'
   source_nonlinear_dis_path = f'/~/Nonlinear/LH{number}/4/dis.npy'

   dest_data_dir = f'/user_data/ajliang/Quijote'
   dest_linear_dis_path = f'/lab_data/aartilab/ajliang/Quijote/LH{number}/lin.npy'
   dest_params_path = f'/lab_data/aartilab/ajliang/Quijote/LH{number}/params.npy'
   dest_nonlinear_dis_path = f'/lab_data/aartilab/ajliang/Quijote/LH{number}/nonlin.npy'

   tdata.add_item(
      source_path=source_linear_dis_path,
      destination_path=dest_linear_dis_path,
   )
   tdata.add_item(
      source_path=source_params_path,
      destination_path=dest_params_path,
   )
   tdata.add_item(
      source_path=source_nonlinear_dis_path,
      destination_path=dest_nonlinear_dis_path,
   )


def download_all(tc: globus_sdk.TransferClient, tdata: globus_sdk.TransferData):
   for i in range(2000):
      download_one(number=i, tdata=tdata)


if __name__ == '__main__':

   source_endpoint_id = '9da966a0-58ec-11ed-89dc-ede5bae4f491'
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
                                 label="Download all Quijote data",
                                 sync_level="checksum")

   download_all(tc=tc, tdata=tdata)

   transfer_result = tc.submit_transfer(tdata)
   print("task_id =", transfer_result["task_id"])

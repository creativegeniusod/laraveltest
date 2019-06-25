<?php
namespace App\Http\Controllers;
use Illuminate\Http\Request;
use App\Helper\Reply;
use Response;
use App\Http\Requests\Media\StoreMedia;
use Illuminate\Support\Facades\Validator;
use Illuminate\Support\Facades\Input;

class MediaController extends Controller
{
    public function index() {
        $attachments_path = public_path('user-uploads/attachments/');
        $pdf_path = public_path('user-uploads/pdf/');
        $python_script = base_path('my_script.py');
        $file = '5cfad3d56a98e.jpg';
        $pdf = '5cfad3d56a98e.pdf';
        $cmd = 'python '.$python_script. ' "'.$attachments_path.$file.'" "'.$file.'" "'.$pdf_path.$pdf.'"';
        $output = shell_exec($cmd);
    }

    function uploadImage(Request $request){
        // Setup the validator
        $rules = array('image' => 'required|image|mimes:jpeg,png,jpg,gif|max:2048');
        $validator = Validator::make(Input::all(), $rules);

        // Validate the input and return correct response
        if ($validator->fails()){
          return Response::json(array(
              'success' => false,
              'errors' => $validator->getMessageBag()->toArray()

          ), 400); // 400 being the HTTP code for an invalid request.
        }

        /****** image ******/
        if ($request->hasFile('image')) {
          $extension = $request->file('image')->getClientOriginalExtension();
          $imageName = $request->file('image')->getClientOriginalName();
          $filename = basename($imageName,".".$extension).'_'.date("Y_m_d").'_output';

          $dateFormat = date("Y_m_d").'_output';
          $file = $request->file('image')->move(config('constants.upload_path.pythonscript'), $filename.".".$extension);
          $attachments_path = public_path('user-uploads/pythonscript/');
          $pdf_path = public_path('user-uploads/pythonscript/');
          $originam_path = public_path('user-uploads/original/');
          $temp_pdf_path = public_path('user-uploads/pdf/');
          $file = $filename.".".$extension;
          $pdf = $filename.'.pdf';
          $python_script = base_path('newContours2.py');
          $tempName = $string = pathinfo($imageName, PATHINFO_FILENAME);  
          //$tempName = str_replace(' ', '', $tempName);         
          $cmd = 'python3 '.$python_script. ' "'.$file.'" "'.$dateFormat.'"  "'.$tempName.'"';
          $output = shell_exec($cmd);
          if($output) {
              $pdf_file = config('constants.upload_path.pythonscript').$pdf;
              return Reply::dataOnly(['status' => 'success',
                  'file' => $file, 'pdf_file' => $pdf_file]);
          }
        }
    }
}

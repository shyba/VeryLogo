use aes::Aes128;
use cipher::generic_array::GenericArray;
use cipher::{BlockEncrypt, KeyInit};

pub type Block = GenericArray<u8, cipher::consts::U16>;

pub fn fixed_key() -> [u8; 16] {
    [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
        0x0C, 0x0D, 0x0E, 0x0F,
    ]
}

pub fn encrypt_blocks_fixed_key(blocks: &mut [Block]) {
    let key = fixed_key();
    let cipher = Aes128::new(GenericArray::from_slice(&key));
    cipher.encrypt_blocks(blocks);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vector() {
        let key = fixed_key();
        let cipher = Aes128::new(GenericArray::from_slice(&key));
        let mut block = Block::clone_from_slice(&[
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB,
            0xCC, 0xDD, 0xEE, 0xFF,
        ]);
        cipher.encrypt_block(&mut block);
        let exp = [
            0x69, 0xC4, 0xE0, 0xD8, 0x6A, 0x7B, 0x04, 0x30, 0xD8, 0xCD, 0xB7, 0x80,
            0x70, 0xB4, 0xC5, 0x5A,
        ];
        assert_eq!(block.as_slice(), &exp);
    }
}

